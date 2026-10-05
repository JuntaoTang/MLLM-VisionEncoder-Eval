import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from vision_encoder_eval.workers.tokbench.tokenzier_vae_scripts.image_scripts.resize_rec import restore_original, smart_padding


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_TOKLIP_DIR = os.path.join(DEFAULT_MODEL_ZOO, "TokLIP")

MODEL_CONFIGS = {
    "toklip_s": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_S_256.pt"),
        "toklip_model_config": "ViT-SO400M-16-SigLIP2-256-toklip",
        "image_size": 256,
    },
    "toklip_l": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_L_384.pt"),
        "toklip_model_config": "ViT-SO400M-16-SigLIP2-384-toklip",
        "image_size": 384,
    },
    "toklip_s_semantic_nn": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_S_256.pt"),
        "toklip_model_config": "ViT-SO400M-16-SigLIP2-256-toklip",
        "image_size": 256,
    },
    "toklip_l_semantic_nn": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_L_384.pt"),
        "toklip_model_config": "ViT-SO400M-16-SigLIP2-384-toklip",
        "image_size": 384,
    },
}


def get_args_parser(default_model_name):
    parser = argparse.ArgumentParser("TokLIP VQ reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(DEFAULT_RECON_ROOT, default_model_name, "text_data", "ic13"),
    )
    parser.add_argument("--model_name", type=str, default=default_model_name, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--toklip_path", type=str, default=os.path.join(SCRIPT_DIR, "TokLIP"))
    parser.add_argument("--toklip_ckpt_path", type=str, default=None)
    parser.add_argument("--vq_ckpt_path", type=str, default=os.path.join(DEFAULT_TOKLIP_DIR, "vq_ds16_t2i.pt"))
    parser.add_argument("--toklip_model_config", type=str, default=None)
    parser.add_argument("--latent_source", type=str, default="quantized", choices=["quantized", "semantic_nn"])
    parser.add_argument("--padding_size", type=int, default=256)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_toklip_vq(args, device):
    toklip_path = os.path.abspath(args.toklip_path)
    toklip_src = os.path.join(toklip_path, "src")
    tokenizer_path = os.path.join(toklip_src, "tokenizer")
    vq_ckpt_path = os.path.abspath(args.vq_ckpt_path)

    # TokLIP-S/L share this VQ tokenizer. Their semantic-encoder checkpoints
    # are irrelevant for the quantized reconstruction used by TokBench.
    require_path(tokenizer_path, "TokLIP tokenizer source directory")
    require_path(vq_ckpt_path, "TokLIP LlamaGen VQ checkpoint")

    sys.path.insert(0, toklip_src)
    from tokenizer.vq_model import VQ_models

    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device)
    vq_model.eval()

    checkpoint = torch.load(vq_ckpt_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise ValueError(f"Could not find VQ weights in checkpoint: {vq_ckpt_path}")
    vq_model.load_state_dict(model_weight)
    return vq_model


def ensure_toklip_vq_checkpoint(toklip_src, vq_ckpt_path):
    pretrained_dir = os.path.join(toklip_src, "tokenizer", "pretrained_models")
    os.makedirs(pretrained_dir, exist_ok=True)
    expected_path = os.path.join(pretrained_dir, "vq_ds16_t2i.pt")
    if os.path.exists(expected_path):
        return expected_path
    os.symlink(vq_ckpt_path, expected_path)
    return expected_path


def load_toklip_semantic_model(args, device):
    toklip_path = os.path.abspath(args.toklip_path)
    toklip_src = os.path.join(toklip_path, "src")
    toklip_ckpt_path = os.path.abspath(args.toklip_ckpt_path or MODEL_CONFIGS[args.model_name]["toklip_ckpt"])
    vq_ckpt_path = os.path.abspath(args.vq_ckpt_path)
    model_config = args.toklip_model_config or MODEL_CONFIGS[args.model_name]["toklip_model_config"]
    image_size = MODEL_CONFIGS[args.model_name]["image_size"]

    require_path(os.path.join(toklip_src, "create_toklip.py"), "TokLIP source directory")
    require_path(toklip_ckpt_path, f"{args.model_name} TokLIP checkpoint")
    require_path(vq_ckpt_path, "TokLIP LlamaGen VQ checkpoint")
    ensure_toklip_vq_checkpoint(toklip_src, vq_ckpt_path)

    sys.path.insert(0, toklip_src)
    from create_toklip import create_toklip

    old_cwd = os.getcwd()
    try:
        os.chdir(toklip_src)
        model, _, _ = create_toklip(
            model=model_config,
            image_size=image_size,
            model_path=toklip_ckpt_path,
            device=device,
        )
    finally:
        os.chdir(old_cwd)

    model.eval()
    return model.visual.trunk


def _toklip_attn_mask(trunk, tokens):
    if not getattr(trunk, "vit_causal_mask", False):
        return None
    num_tokens = tokens.shape[1]
    mask = torch.empty(num_tokens, num_tokens, device=tokens.device)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask.type_as(tokens)


def encode_toklip_semantic_tokens(trunk, image):
    quant, _, _ = trunk.vq.encode(image)
    quant = quant.reshape(quant.shape[0], quant.shape[1], -1).permute(0, 2, 1)
    quant = quant.to(image.dtype)

    tokens = trunk.vq_table(quant)
    tokens = trunk.vq_act(tokens)
    tokens = trunk.vq_mapping(tokens)

    tokens = trunk._pos_embed(tokens)
    tokens = trunk.patch_drop(tokens)
    tokens = trunk.norm_pre(tokens)

    attn_mask = _toklip_attn_mask(trunk, tokens)
    for block in trunk.blocks:
        tokens = block(tokens, attn_mask=attn_mask)
    return trunk.norm(tokens)


def semantic_tokens_to_quant(trunk, tokens, height, width):
    codebook = trunk.vq.quantize.embedding.weight
    if trunk.vq.quantize.l2_norm:
        codebook = torch.nn.functional.normalize(codebook, p=2, dim=-1)
    codebook = codebook.to(device=tokens.device, dtype=tokens.dtype)

    semantic_codebook = trunk.vq_table(codebook)
    semantic_codebook = trunk.vq_act(semantic_codebook)
    semantic_codebook = trunk.vq_mapping(semantic_codebook)

    tokens = torch.nn.functional.normalize(tokens.to(torch.float32), p=2, dim=-1)
    semantic_codebook = torch.nn.functional.normalize(semantic_codebook.to(torch.float32), p=2, dim=-1)
    indices = torch.matmul(tokens, semantic_codebook.t()).argmax(dim=-1)
    return trunk.vq.quantize.get_codebook_entry(
        indices.reshape(-1),
        shape=(tokens.shape[0], codebook.shape[-1], height, width),
        channel_first=True,
    )


def reconstruct_batch(model, input_img, latent_source):
    if latent_source == "quantized":
        quant, _, _ = model.encode(input_img)
        return model.decode(quant)

    if latent_source == "semantic_nn":
        tokens = encode_toklip_semantic_tokens(model, input_img)
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError(f"TokLIP semantic token count is not square: {tokens.shape[1]}")
        quant = semantic_tokens_to_quant(model, tokens, side, side)
        return model.vq.decode(quant)

    raise ValueError(f"Unsupported latent_source: {latent_source}")


def tensor_to_pilimg(img):
    img = torch.clamp(127.5 * img + 128.0, 0, 255)
    img = img.permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(img)


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    if args.latent_source == "quantized":
        model = load_toklip_vq(args, device)
    else:
        model = load_toklip_semantic_model(args, device)
        expected_size = MODEL_CONFIGS[args.model_name]["image_size"]
        if args.padding_size != expected_size:
            raise ValueError(
                f"{args.latent_source} for {args.model_name} requires padding_size={expected_size}, "
                f"got {args.padding_size}"
            )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        inputs = []
        metas = []
        names = []
        for filename in chunk:
            image_path = os.path.join(args.image_path, filename)
            original_img = Image.open(image_path).convert("RGB")
            padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))
            inputs.append(transform(padded_img))
            metas.append(meta)
            names.append(filename)

        input_img = torch.stack(inputs, dim=0).to(device)
        with torch.no_grad():
            samples = reconstruct_batch(model, input_img, args.latent_source)

        for sample, meta, filename in zip(samples, metas, names):
            rec_img = tensor_to_pilimg(sample)
            final_img = restore_original(rec_img, meta)
            final_img.save(os.path.join(image_save_pth, filename))

    print(args.chunk_idx, " is done")
