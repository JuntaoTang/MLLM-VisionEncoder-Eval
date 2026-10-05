# -*- coding: utf-8 -*-
"""
extract_discrete_features.py
===================================
Extract continuous representations for the discrete tokenizers by reusing their
upstream wrappers, e.g.:

  vilau_256    : VilaUTokenizer.encode_post_quant_features
                 (RQ-quantized patch features, 1024-d)
  toklip_*     : TokLIPTokenizer.encode_post_quant_features
                 (ViT patch features after VQ, 1152-d)
  unitok_attn  : UniTokTokenizer.encode (post-quant features via
                 quant_proj + quantizer + post_quant_proj)

`--list` prints every encoder this script can extract (the `SPECS` table); the
study uses a subset of them.

One mean-pooled + L2-normalized vector per image, saved in
features_diverse layout, row-aligned via image_paths.txt (unreadable
images zero-filled, never shifted).

Usage (from the CKA-X/ root):
    python scripts/extract_discrete_features.py --list
    python scripts/extract_discrete_features.py --dry_run --tokenizers vilau_256
    nohup python scripts/extract_discrete_features.py > discrete_extract.log 2>&1 &
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_encoder_eval.workers.ckax.scripts.ckax_common import resolve_data_path  # noqa: E402
from vision_encoder_eval.core.runtime import asset_path

# Roots of the upstream tree and of the discrete checkpoints; override with
# the environment variables VTB_ROOT / DISC_CKPT.  VTB_ROOT must contain the
# wrappers at src/discrete/model/tokenizers/... ; DISC_CKPT holds the
# checkpoints (vilau/ toklip/ unitok/ uniar/ ...).
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTB_ROOT = os.environ.get("VTB_ROOT", asset_path('mllm'))
TOKLIP_ROOT = asset_path('package', 'mllm/discrete/model/tokenizers/toklip')
DISC_CKPT = os.environ.get("DISC_CKPT",
                           asset_path('download', 'tokenizer/discrete'))
# VQGAN weights for the TokLIP wrapper; VQGAN_PATH (if set) is tried first.
# An offline machine must have this file locally.
VQGAN_CANDIDATES = [
    os.environ.get("VQGAN_PATH", ""),
    os.path.join(DISC_CKPT, "toklip", "vq_ds16_t2i.pt"),
]
VQGAN = next((p for p in VQGAN_CANDIDATES if p and os.path.isfile(p)),
             os.path.join(DISC_CKPT, "toklip", "vq_ds16_t2i.pt"))

SPECS = {
    "vilau_256": dict(
        kind="vilau",
        path=os.path.join(DISC_CKPT, "vilau", "vila-u-7b-256",
                          "vision_tower"),
        image_size=256,
    ),
    "toklip_s_256": dict(
        kind="toklip",
        path=os.path.join(DISC_CKPT, "toklip", "TokLIP_S_256.pt"),
        model_config="ViT-SO400M-16-SigLIP2-256-toklip",
        image_size=256, vqgan=VQGAN,
    ),
    "toklip_l_384": dict(
        kind="toklip",
        path=os.path.join(DISC_CKPT, "toklip", "TokLIP_L_384.pt"),
        model_config="ViT-SO400M-16-SigLIP2-384-toklip",
        image_size=384, vqgan=VQGAN,
    ),
    "unitok_attn": dict(
        kind="unitok",
        path=os.path.join(DISC_CKPT, "unitok", "unitok_tokenizer.pth"),
        num_query=256, image_size=256,
    ),
    "uniar_bsq": dict(
        kind="uniar",
        path=os.path.join(DISC_CKPT, "uniar"),  # dir containing bsq_encoder/
        image_size=512,
    ),
}


def ensure_vtb_path():
    if VTB_ROOT not in sys.path:
        sys.path.insert(0, VTB_ROOT)


def _ensure_timm_local_data_stub():
    """The toklip fork's timm_local copy lacks the data/ subpackage.
    timm_local model files (beit.py, ...) import normalization
    constants from it at module import time, which breaks the whole
    `from timm_local import *` chain in timm_model_toklip.py
    (create_model silently goes missing -> NameError at model build).
    Fix: register a stub for timm_local.data carrying the REAL timm
    constant values BEFORE the package import, so the chain succeeds
    and the genuine timm_local package (with create_model) loads.
    Anything else requested from timm_local.data gets a permissive
    placeholder (training-only plumbing, unused at inference)."""
    import importlib
    import types
    if TOKLIP_ROOT not in sys.path:
        sys.path.insert(0, TOKLIP_ROOT)
    if "timm_local.data" in sys.modules:
        return
    try:
        importlib.import_module("timm_local.data")
        return  # real module present, nothing to do
    except Exception:
        pass
    # drop partially-initialized timm_local modules from the failed import
    for k in [k for k in sys.modules
              if k == "timm_local" or k.startswith("timm_local.")]:
        del sys.modules[k]

    class _Anything:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            return _Anything()

    data = types.ModuleType("timm_local.data")
    data.__path__ = []
    # A REAL STRING __file__ is required.  torch.library.register_fake (run
    # when torchvision is imported for the first time, via
    # torchvision/_meta_registrations.py) calls inspect.getabsfile() on every
    # module in sys.modules.  Without __file__, the module-level __getattr__
    # below would hand back the _Anything CLASS for '__file__', so inspect
    # ends up calling .endswith() on a type object:
    #   AttributeError: type object '_Anything' has no attribute 'endswith'
    # This only bites when toklip is the FIRST tokenizer in the process
    # (torchvision not yet imported); if another tokenizer already imported
    # torchvision, _meta_registrations never re-runs and the bug is invisible.
    data.__file__ = "<stub timm_local.data>"
    # real timm.data.constants values
    data.IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    data.IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    data.IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    data.IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    data.OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    data.OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def _stub_getattr(name):
        # never satisfy dunder lookups (inspect/os ask for __file__,
        # __spec__, __loader__, ...) -- raise so callers see a normal
        # missing-attribute and use their fallback path.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Anything

    data.__getattr__ = _stub_getattr  # module-level fallback
    sys.modules["timm_local.data"] = data
    print("  [INFO] stubbed missing module timm_local.data with real "
          "constants (inference-only workaround)")


def _patch_unitok_geglumlp():
    """Make the UniTok fork's `GeGluMlp` tolerant of newer timm kwargs.

    The upstream UniTok checkout (``$VTB_ROOT``) registers a custom
    ``vitamin_large`` whose ``GeGluMlp.__init__`` predates the timm installed
    here.  Newer timm's ``Block`` forwards ``norm_layer`` (and possibly other
    kwargs) to the MLP factory, which the fork's MLP does not declare, so
    ``timm.create_model`` dies with

        TypeError: GeGluMlp.__init__() got an unexpected keyword argument
                   'norm_layer'

    The fix is applied at runtime rather than by editing the vendored repo: wrap
    ``GeGluMlp.__init__`` so that any kwarg not present in its own signature is
    dropped.  Must run AFTER the wrapper import (which pulls vitamin in and
    registers the model) and BEFORE ``timm.create_model`` is called.
    """
    import inspect
    targets = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        f = (getattr(mod, "__file__", None) or "").replace("\\", "/")
        G = getattr(mod, "GeGluMlp", None)
        if isinstance(G, type):
            targets.append((0 if "UniTok" in f else 1, name, f, G))
    if not targets:
        print("  [unitok-patch] GeGluMlp not found in sys.modules; "
              "nothing patched")
        return False
    targets.sort(key=lambda t: t[0])
    _, name, f, G = targets[0]
    if getattr(G, "_kwarg_tolerant_patch", False):
        return True
    orig = G.__init__
    ok = set(inspect.signature(orig).parameters)
    def _init(self, *args, **kwargs):
        for k in [k for k in kwargs if k not in ok]:
            kwargs.pop(k)
        return orig(self, *args, **kwargs)
    G.__init__ = _init
    G._kwarg_tolerant_patch = True
    print(f"  [unitok-patch] GeGluMlp.__init__ in '{name}' ({f}) now drops "
          f"kwargs outside its signature {sorted(ok)}")
    return True


def load_encoder(tok_id, spec, device):
    """Returns encode_fn(list[PIL]) -> (B, N_tokens, D) float tensor."""
    ensure_vtb_path()
    kind = spec["kind"]

    if kind == "vilau":
        from vision_encoder_eval.mllm.discrete.model.tokenizers.vilau.wrapper import (
            VilaUTokenizer,
        )
        tok = VilaUTokenizer.from_checkpoint(spec["path"])
        tok._model = tok._model.to(device)

        def encode(imgs):
            # NOTE: pass a preprocessed TENSOR, not a PIL list:
            # wrapper's encode_post_quant_features ends with
            # feats.to(pixel_values.dtype), which crashes on lists.
            x = tok.preprocess(imgs)
            with torch.no_grad():
                f = tok.encode_post_quant_features(x)
            return f.float()
        return encode, tok

    if kind == "toklip":
        if not os.path.isfile(spec["vqgan"]):
            raise RuntimeError(
                "TokLIP VQGAN weights not found at %s.  The wrapper would "
                "try to DOWNLOAD them and fail with 'urlopen error [Errno "
                "99]' on this offline server.  Known copies: %s"
                % (spec["vqgan"], VQGAN_CANDIDATES))
        _ensure_timm_local_data_stub()
        from vision_encoder_eval.mllm.discrete.model.tokenizers.toklip.wrapper import (
            TokLIPTokenizer,
        )
        tok = TokLIPTokenizer.from_checkpoint(
            spec["path"], model_config=spec["model_config"],
            image_size=spec["image_size"],
            vqgan_checkpoint=spec["vqgan"])
        tok._visual = tok._visual.to(device)

        def encode(imgs):
            with torch.no_grad():
                f = tok.encode_post_quant_features(imgs)
            return f.float()
        return encode, tok

    if kind == "unitok":
        from torchvision import transforms
        from vision_encoder_eval.mllm.discrete.model.tokenizers.unitok.wrapper import (
            UniTokTokenizer,
        )
        _patch_unitok_geglumlp()      # timm/newer-kwarg compatibility
        tok = UniTokTokenizer(spec["path"],
                              num_query=spec.get("num_query"))
        tok = tok.to(device)          # module holds encoder+quant parts
        to_tensor = transforms.Compose([
            transforms.Resize((spec["image_size"], spec["image_size"]),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),    # [0,1]; wrapper maps to [-1,1]
        ])

        def encode(imgs):
            # wrapper API: encode(tensor [0,1]) -> post-quant features
            # (encoder -> quant_proj -> quantizer -> post_quant_proj)
            x = torch.stack([to_tensor(im) for im in imgs])
            with torch.no_grad():
                f = tok.encode(x)
            return f.float()
        return encode, tok

    if kind == "uniar":
        from torchvision import transforms
        from vision_encoder_eval.mllm.discrete.model.tokenizers.uniar.wrapper import (
            UniARTokenizer,
        )
        tok = UniARTokenizer.from_checkpoint(
            spec["path"], image_size=spec["image_size"], bsq_only=True,
            use_deepstack=True, fp16=True)
        tok = tok.to(device)
        to_tensor = transforms.Compose([
            transforms.Resize((spec["image_size"], spec["image_size"]),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        def encode(imgs):
            x = torch.stack([to_tensor(im) for im in imgs])
            with torch.no_grad():
                f = tok.encode_post_quant_features(x)  # [B, N, hidden*levels]
            return f.float()
        return encode, tok

    raise ValueError(f"unknown kind {kind}")


def resolve_image_list(args):
    if not args.align_to or not os.path.exists(args.align_to):
        raise SystemExit(
            f"  [ERROR] alignment list not found: {args.align_to}\n"
            "  pass --align_to <features_diverse>/<tok>/image_paths.txt")
    with open(args.align_to, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    fixed, n_remap = [], 0
    for l in lines:
        if os.path.exists(l):
            fixed.append(l)
        else:
            fixed.append(os.path.join(args.image_dir, os.path.basename(l)))
            n_remap += 1
    if n_remap:
        print(f"  [WARN] remapped {n_remap} missing paths into "
              f"{args.image_dir}")
    missing = [p for p in fixed if not os.path.exists(p)]
    if missing:
        print(f"  [WARN] {len(missing)} images still missing "
              f"(will zero-fill): " + ", ".join(missing[:3]))
    if args.num_images and args.num_images < len(fixed):
        fixed = fixed[:args.num_images]
    return fixed


def extract_one(tok_id, args, image_paths):
    spec = SPECS[tok_id]
    if not os.path.exists(spec["path"]):
        print(f"  [{tok_id}] SKIP: weights not found at {spec['path']}")
        return {"tokenizer": tok_id, "status": "missing_weights"}

    out_dir = os.path.join(args.output_dir, tok_id)
    feat_path = os.path.join(out_dir, "visual_features.pt")
    if not args.force and os.path.exists(feat_path):
        try:
            ex = torch.load(feat_path, map_location="cpu",
                            weights_only=False)
            if ex.shape[0] >= len(image_paths):
                print(f"  [{tok_id}] CACHED "
                      f"({ex.shape[0]} x {ex.shape[1]})")
                return {"tokenizer": tok_id, "status": "cached"}
        except Exception:
            pass

    device = args.device
    encode, tok = load_encoder(tok_id, spec, device)

    first_imgs = []
    for p in image_paths[:64]:
        try:
            first_imgs.append(Image.open(p).convert("RGB"))
            if len(first_imgs) >= 2:
                break
        except Exception:
            continue
    if not first_imgs:
        print(f"  [{tok_id}] FAILED: no readable images")
        return {"tokenizer": tok_id, "status": "failed",
                "error": "no readable images"}
    f0 = encode(first_imgs)
    if f0.dim() == 2:
        f0 = f0.unsqueeze(1)
    dim_out = f0.shape[-1]
    print(f"  [{tok_id}] tokens {tuple(f0.shape)} -> pooled {dim_out}-d")
    if args.dry_run:
        print(f"  [{tok_id}] dry_run OK; nothing saved")
        return {"tokenizer": tok_id, "status": "dry_run"}

    all_feats = torch.zeros(len(image_paths), dim_out)
    n_bad = 0
    bs = args.batch_size
    for i in tqdm(range(0, len(image_paths), bs),
                  desc=f"  {tok_id}", unit="batch"):
        bpaths = image_paths[i:i + bs]
        imgs, okpos = [], []
        for j, p in enumerate(bpaths):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                okpos.append(j)
            except Exception:
                n_bad += 1
        if imgs:
            out = encode(imgs)
            if out.dim() == 2:
                out = out.unsqueeze(1)
            out = F.normalize(out.mean(dim=1).float(), dim=-1)
            for r, j in enumerate(okpos):
                all_feats[i + j] = out[r].cpu()

    del tok, encode
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_feats, feat_path)
    meta = {
        "tokenizer": tok_id, "dim": dim_out, "n": all_feats.shape[0],
        "space": "discrete_post_quant",
        "kind": spec["kind"], "weights": spec["path"],
        "aggregation": "mean over post-quant tokens, L2-normalized",
        "n_zero_filled": n_bad,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "image_paths.txt"), "w") as f:
        f.write("\n".join(image_paths))
    print(f"  [{tok_id}] saved {tuple(all_feats.shape)} -> {feat_path} "
          f"(zero-filled {n_bad})")
    return {"tokenizer": tok_id, "status": "ok", "dim": dim_out}


def do_list(args):
    print(f"\n  Discrete tokenizer weights:")
    print(f"  {'tokenizer':<16} {'kind':<8} {'image':>5}  path exists")
    for tok_id, spec in sorted(SPECS.items()):
        ex = "yes" if os.path.exists(spec["path"]) else "NO"
        print(f"  {tok_id:<16} {spec['kind']:<8} "
              f"{spec['image_size']:>5}  {ex}  {spec['path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--align_to", type=str, default=None)
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--tokenizers", nargs="+", default=None)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        do_list(args)
        return

    diverse = resolve_data_path("features_diverse")
    if args.output_dir is None:
        args.output_dir = diverse
    if args.align_to is None:
        cand = os.path.join(diverse, "clip_openai__l14",
                            "image_paths.txt")
        if not os.path.exists(cand):
            for td in sorted(Path(diverse).iterdir()):
                p = td / "image_paths.txt"
                if p.exists():
                    cand = str(p)
                    break
        args.align_to = cand
    if args.image_dir is None:
        args.image_dir = resolve_data_path("images_diverse")

    image_paths = resolve_image_list(args)
    print(f"\n  Image list: {len(image_paths)} images "
          f"(aligned via {args.align_to})")

    toks = args.tokenizers or sorted(SPECS)
    results = []
    for tok_id in toks:
        if tok_id not in SPECS:
            print(f"  [{tok_id}] SKIP: unknown tokenizer")
            continue
        try:
            r = extract_one(tok_id, args, image_paths)
        except Exception as e:
            import traceback
            print(f"  [{tok_id}] FAILED: {e}")
            print(traceback.format_exc())
            r = {"tokenizer": tok_id, "status": "failed",
                 "error": str(e)}
        if r:
            results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n  Summary:")
    for r in results:
        print(f"    {r['tokenizer']:<16} {r['status']}")


if __name__ == "__main__":
    main()
