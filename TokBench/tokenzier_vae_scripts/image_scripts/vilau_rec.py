import argparse
import importlib
import importlib.machinery
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_MODEL_ZOO, "VILA-U", "vila-u-7b-256")
DEFAULT_SIGLIP_CONFIG = os.path.join(DEFAULT_MODEL_ZOO, "VILA-U", "siglip-large-patch16-256")


def get_args_parser():
    parser = argparse.ArgumentParser("VILA-U vision tokenizer reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "vilau_7b_256", "text_data", "ic13"))
    parser.add_argument("--model_name", type=str, default="vilau_7b_256")
    parser.add_argument("--vilau_path", type=str, default=os.path.join(SCRIPT_DIR, "vila-u"))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--siglip_config_path", type=str, default=DEFAULT_SIGLIP_CONFIG)
    parser.add_argument("--padding_size", type=int, default=256)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def torch_dtype(dtype_name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_vision_tower_path(model_path):
    vision_tower_path = os.path.join(model_path, "vision_tower")
    if os.path.isdir(vision_tower_path):
        return vision_tower_path
    return model_path


def import_vilau_vision_classes(vilau_path):
    """Import only VILA-U's vision tokenizer without eager LLM dependencies.

    The upstream ``vila_u`` and ``vila_u.model`` package initializers eagerly
    import the chat/LLM stack.  Tokenizer-only inference does not need that
    stack, and importing it introduces unrelated dependencies such as loguru
    plus APIs removed by newer Transformers releases.  Lightweight package
    shells preserve normal relative imports inside the vision subtree while
    skipping those two initializers.
    """
    package_paths = {
        "vila_u": os.path.join(vilau_path, "vila_u"),
        "vila_u.model": os.path.join(vilau_path, "vila_u", "model"),
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [package_path]
        package.__package__ = package_name
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package

    module = importlib.import_module(
        "vila_u.model.multimodal_encoder.rqvaesigliptransformer"
    )
    return module.RQVAESIGLIPTransformer, module.RQVAESIGLIPTransformerConfig


def restore_position_ids(model):
    """Rebuild non-persistent position IDs after Transformers meta loading.

    Transformers 5.x materializes non-persistent buffers from a meta model with
    ``empty_like``.  Since position IDs are deliberately absent from the
    checkpoint, that leaves them as uninitialized integers.  Recreate them
    from each position embedding before moving the model to its target device.
    """
    restored = 0
    for module in model.modules():
        position_embedding = getattr(module, "position_embedding", None)
        position_ids = getattr(module, "position_ids", None)
        if not isinstance(position_embedding, torch.nn.Embedding) or not torch.is_tensor(
            position_ids
        ):
            continue
        count = int(position_embedding.num_embeddings)
        values = torch.arange(count, device=position_embedding.weight.device).expand(1, -1)
        module.register_buffer("position_ids", values, persistent=False)
        restored += 1
    if restored == 0:
        raise RuntimeError("VILA-U model has no position_ids buffers to restore")


def load_vilau_tokenizer(args, device):
    vilau_path = os.path.abspath(args.vilau_path)
    model_path = os.path.abspath(args.model_path)
    siglip_config_path = os.path.abspath(args.siglip_config_path)
    vision_tower_path = resolve_vision_tower_path(model_path)

    require_path(os.path.join(vilau_path, "vila_u"), "VILA-U source directory")
    require_path(vision_tower_path, "VILA-U vision tower checkpoint")
    require_path(os.path.join(siglip_config_path, "config.json"), "SigLIP base config")

    RQVAESIGLIPTransformer, RQVAESIGLIPTransformerConfig = import_vilau_vision_classes(
        vilau_path
    )
    from transformers import CLIPImageProcessor

    dtype = torch_dtype(args.dtype)
    config = RQVAESIGLIPTransformerConfig.from_pretrained(vision_tower_path)
    config.rqvaesiglip["pretrained_model"] = siglip_config_path
    model = RQVAESIGLIPTransformer.from_pretrained(
        vision_tower_path,
        config=config,
        dtype=dtype,
    )
    restore_position_ids(model)
    model.to(device=device, dtype=dtype)
    model.eval()

    if config.hidden_size == 1152:
        image_size = 384
    elif config.hidden_size == 1024:
        image_size = 256
    else:
        raise NotImplementedError(f"Unsupported VILA-U hidden size: {config.hidden_size}")

    image_processor = CLIPImageProcessor(
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    return SimpleNamespace(model=model.rqvaesiglip, image_processor=image_processor, dtype=dtype, image_size=image_size)


def decode_to_pil(decoded):
    image = decoded.to(torch.float32).add(1).mul(127.5).clamp(0, 255)
    image = image.permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(image)


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    tokenizer = load_vilau_tokenizer(args, device)

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
            pixel_values = tokenizer.image_processor.preprocess(padded_img, return_tensors="pt")["pixel_values"][0]
            inputs.append(pixel_values)
            metas.append(meta)
            names.append(filename)

        input_img = torch.stack(inputs, dim=0).to(device=device, dtype=tokenizer.dtype)
        with torch.no_grad():
            _, z_q = tokenizer.model.encode_image(input_img)
            samples = tokenizer.model.decode(z_q)

        for sample, meta, filename in zip(samples, metas, names):
            rec_img = decode_to_pil(sample)
            if rec_img.size != (args.padding_size, args.padding_size):
                rec_img = rec_img.resize((args.padding_size, args.padding_size), Image.LANCZOS)
            final_img = restore_original(rec_img, meta)
            final_img.save(os.path.join(image_save_pth, filename))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
