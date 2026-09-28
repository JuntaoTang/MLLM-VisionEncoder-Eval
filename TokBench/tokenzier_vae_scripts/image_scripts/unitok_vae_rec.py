import argparse
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")


def get_args_parser():
    parser = argparse.ArgumentParser("UniTok reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "unitok", "text_data", "ic13"))
    parser.add_argument("--unitok_path", type=str, default=os.path.join(SCRIPT_DIR, "UniTok"))
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(DEFAULT_MODEL_ZOO, "unitok_20250227", "unitok_tokenizer.pth"),
    )
    parser.add_argument("--padding_size", type=int, default=1024)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def tensor_to_pilimgs(img: torch.Tensor):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in img]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def patch_unitok_timm_compat():
    import models.vitamin as vitamin

    if "device" not in inspect.signature(vitamin.HybridEmbed.__init__).parameters:
        old_hybrid_init = vitamin.HybridEmbed.__init__

        def hybrid_init(self, *args, device=None, dtype=None, **kwargs):
            old_hybrid_init(self, *args, **kwargs)
            factory_kwargs = {k: v for k, v in {"device": device, "dtype": dtype}.items() if v is not None}
            if factory_kwargs:
                self.to(**factory_kwargs)

        vitamin.HybridEmbed.__init__ = hybrid_init

    if "device" not in inspect.signature(vitamin.GeGluMlp.__init__).parameters:
        old_geglu_init = vitamin.GeGluMlp.__init__

        def geglu_init(
            self,
            in_features,
            hidden_features,
            act_layer=None,
            drop=0.0,
            norm_layer=None,
            bias=True,
            device=None,
            dtype=None,
            **kwargs,
        ):
            del norm_layer, bias, kwargs
            old_geglu_init(self, in_features, hidden_features, act_layer=act_layer, drop=drop)
            factory_kwargs = {k: v for k, v in {"device": device, "dtype": dtype}.items() if v is not None}
            if factory_kwargs:
                self.to(**factory_kwargs)

        vitamin.GeGluMlp.__init__ = geglu_init


def patch_unitok_legacy_causal_projection():
    import models.vqvae as vqvae

    attn_cls = getattr(vqvae, "PlainAttention", None) or getattr(vqvae, "CausalAttention", None)
    if attn_cls is None:
        raise RuntimeError("Cannot find UniTok projection attention class.")

    def causal_forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = x.shape
        qkv = F.linear(
            input=x,
            weight=self.qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        q, k, v = qkv.reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = F.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(bsz, num_tokens, -1)
        return self.proj(x)

    attn_cls.forward = causal_forward


def load_unitok(args):
    unitok_path = os.path.abspath(args.unitok_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(unitok_path, "UniTok code directory")
    require_path(ckpt_path, "UniTok checkpoint")

    sys.path.insert(0, unitok_path)
    patch_unitok_timm_compat()
    patch_unitok_legacy_causal_projection()
    from models.unitok import UniTok
    from utils.config import Args
    from utils.data import normalize_01_into_pm1

    ckpt = torch.load(ckpt_path, map_location="cpu")
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt["args"])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt["trainer"]["unitok"])
    unitok.to("cuda")
    unitok.eval()

    preprocess = transforms.Compose([transforms.ToTensor(), normalize_01_into_pm1])
    return unitok, preprocess


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    unitok, preprocess = load_unitok(args)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        # Every image is smart_padded to padding_size x padding_size, so the whole
        # chunk shares one spatial size and can be encoded/decoded as a single batch.
        tensors, metas = [], []
        for name in chunk:
            original_img = Image.open(os.path.join(args.image_path, name)).convert("RGB")
            padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))
            tensors.append(preprocess(padded_img))
            metas.append(meta)

        img = torch.stack(tensors).to("cuda")

        with torch.no_grad():
            code_idx = unitok.img_to_idx(img)
            rec_imgs = unitok.idx_to_img(code_idx)

        for name, rec_img, meta in zip(chunk, tensor_to_pilimgs(rec_imgs), metas):
            final_img = restore_original(rec_img, meta)
            final_img.save(os.path.join(image_save_pth, name))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
