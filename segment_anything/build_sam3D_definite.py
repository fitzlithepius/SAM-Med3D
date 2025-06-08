# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from .SAMMed3D import ImageEncoderViT3D
from .maskDecoder import MaskDecoderModule, get_sam_checkpoint
from utils.infer_utils_modified import validate_paired_img_gt

import debugpy

# debugpy.listen(("0.0.0.0", 4544))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print("You can debug your script now")


if __name__ == "__main__":
    # important parameters
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]

    cache_dir = "/shares/menze.dqbm.uzh/chengrun/MultiModalModel"

    # SAM Encoder
    sam_encoder = ImageEncoderViT3D(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # Load the MedSAM checkpoint
    checkpoint = get_sam_checkpoint(cache_dir)
    checkpoint = {
        k.replace("image_encoder.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("image_encoder")
    }

    incompatibleKeys = sam_encoder.load_state_dict(checkpoint, strict=False)
    print(f"Loaded MedSAM checkpoint with incompatible keys {incompatibleKeys}")

    for param in sam_encoder.parameters():
        param.requires_grad = False

    # Create SAM Decoder
    sam_decoder = MaskDecoderModule(
        seg_prompt_embed_dim=prompt_embed_dim,
        embedding_dim=2304,
        number_image_patches=image_embedding_size,
        image_size=128,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        logger=None,
    )
    print(sam_decoder)

    """ 2. read and pre-process your input data """
    img_path = "./test_data/amos_val_toy_data/imagesVa/amos_0013.nii.gz"
    gt_path = "./test_data/amos_val_toy_data/labelsVa/amos_0013.nii.gz"
    out_path = "./test_data/amos_val_toy_data/pred/amos_0013.nii.gz"

    """ 3. infer with the pre-trained SAM-Med3D model """
    print("Validation start! plz wait for some times.")
    validate_paired_img_gt(
        sam_encoder, sam_decoder, img_path, gt_path, out_path, num_clicks=1
    )
    print("Validation finish! plz check your prediction.")
