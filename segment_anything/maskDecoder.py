from typing import Optional
from torch import nn
import torch

import torch.nn.functional as F
from monai.losses import DiceLoss

from .SAMMed3D.mask_decoder3D import MaskDecoder3D
from .SAMMed3D.position_embedding_random import PositionEmbeddingRandom3D
import os
from typing import Dict
import gdown
from argparse import Namespace

torch.serialization.add_safe_globals([Namespace])


class MaskDecoderModule(nn.Module):
    # TODO: Change print back to logger
    def __init__(
        self,
        seg_prompt_embed_dim: int,
        embedding_dim: int,
        number_image_patches: int,
        image_size: int,
        torch_dtype: str,
        cache_dir: str,
        logger,
    ):
        super().__init__()
        self.logger = logger
        self.seg_prompt_embed_dim = seg_prompt_embed_dim
        self.embedding_dim = embedding_dim
        self.number_image_patches = number_image_patches
        self.image_size = image_size

        self.mask_decoder = MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=self.seg_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.mask_decoder.to(dtype=torch_dtype)

        # Load the MedSAM checkpoint
        checkpoint = get_sam_checkpoint(cache_dir)
        checkpoint_decoder = {
            k.replace("mask_decoder.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if k.startswith("mask_decoder")
        }
        incompatibleKeys = self.mask_decoder.load_state_dict(
            checkpoint_decoder, strict=False
        )
        print(f"Incompatible keys for the mask decoder: {incompatibleKeys}")

        self.projection_layer = nn.Sequential(
            nn.Linear(embedding_dim, self.seg_prompt_embed_dim * 2, dtype=torch_dtype),
            nn.ReLU(),
            nn.Linear(
                self.seg_prompt_embed_dim * 2,
                self.seg_prompt_embed_dim,
                dtype=torch_dtype,
            ),
        )

        self.pe_layer = PositionEmbeddingRandom3D(
            self.seg_prompt_embed_dim // 3,
            dtype=torch_dtype,
            device=self.projection_layer[0].weight.device,
        )
        checkpoint_pe_layer = {
            k.replace("prompt_encoder.pe_layer.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if k == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
        }
        incompatibleKeys = self.pe_layer.load_state_dict(
            checkpoint_pe_layer, strict=False
        )
        print(f"Incompatible keys for the pe_layer: {incompatibleKeys}")

        self.no_mask_embed = nn.Embedding(
            1, self.seg_prompt_embed_dim, dtype=torch_dtype
        )
        checkpoint_no_mask_embed = {
            k.replace("prompt_encoder.no_mask_embed.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if k == "prompt_encoder.no_mask_embed.weight"
        }
        incompatibleKeys = self.no_mask_embed.load_state_dict(
            checkpoint_no_mask_embed, strict=False
        )
        print(f"Incompatible keys for the no_mask_embed: {incompatibleKeys}")

        self.seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        # self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # Freeze the mask decoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        original_sizes: int,
        seg_pixel_values: Optional[torch.Tensor] = None,
    ):
        prompt_embeddings_tensor = self.projection_layer(prompt_embeddings)

        bs = len(image_embeddings)
        image_pe = self.pe_layer(
            (
                self.number_image_patches,
                self.number_image_patches,
                self.number_image_patches,
            )
        ).unsqueeze(0)
        dense_prompt_embeddings = self.no_mask_embed.weight.reshape(
            1, -1, 1, 1, 1
        ).expand(
            bs,
            -1,
            self.number_image_patches,
            self.number_image_patches,
            self.number_image_patches,
        )
        # prompt_embeddings_tensor = prompt_embeddings

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings.to(torch.bfloat16),
            image_pe=image_pe,
            sparse_prompt_embeddings=prompt_embeddings_tensor.to(torch.bfloat16),
            dense_prompt_embeddings=dense_prompt_embeddings.to(torch.bfloat16),
            multimask_output=False,
        )

        # BUG: to be fixed
        # masks = [
        #     self.postprocess_masks(
        #         low_res_masks[i].squeeze(0),
        #         input_size=(self.image_size, self.image_size, self.image_size),
        #         original_size=original_sizes[i],
        #     ).unsqueeze(0)
        #     for i in range(len(low_res_masks))
        # ]
        masks = F.interpolate(
            low_res_masks, size=original_sizes[0], mode="trilinear", align_corners=False
        )
        if seg_pixel_values is None:
            return masks, None

        # Calculate the frequency of positive and negative points
        segmentation_losses = []
        for mask, ground_truth_mask in zip(masks, seg_pixel_values):
            total_points = ground_truth_mask.numel()
            positive_points = ground_truth_mask.sum()
            negative_points = total_points - positive_points

            # Calculate the inverse frequency weights
            weight_positive = negative_points / total_points
            weight_negative = positive_points / total_points

            # Create a weight map
            weight_map = (
                ground_truth_mask * weight_positive
                + (1 - ground_truth_mask) * weight_negative
            )

            segmentation_losses.append(
                self.seg_loss(mask, ground_truth_mask)
                + F.binary_cross_entropy_with_logits(
                    mask,
                    ground_truth_mask.float(),
                    reduction="mean",
                    weight=weight_map.float(),
                )
            )

        return masks, torch.stack(segmentation_losses).mean()

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_size, self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1], : input_size[2]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    # def set_external_parameters(self):
    #     parameters_to_register = list(self.parameters())
    #     for parameter_to_register in parameters_to_register:
    #         deepspeed.zero.register_external_parameter(self, parameter_to_register)


def get_sam_checkpoint(cache_dir) -> Dict:

    if not os.path.exists(os.path.join(cache_dir, "sam_med3d_turbo.pth")):
        gdown.download(
            id="1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9",
            output=os.path.join(cache_dir, "sam_med3d_turbo.pth"),
        )

    with open(os.path.join(cache_dir, "sam_med3d_turbo.pth"), "rb") as f:
        state_dict = torch.load(
            f,
            weights_only=True,
            # map_location=torch.device("cpu"),
        )
    return state_dict
