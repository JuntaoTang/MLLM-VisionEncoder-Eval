"""Shared VLMEvalKit monkey-patches used by launcher and local scoring."""

from __future__ import annotations

import os


def patch_subsampled_tsv_md5() -> None:
    """Never re-download local LMUData TSVs when MD5 mismatches.

    Path-based local TSVs (and subsampled caches) will not match upstream MD5.
    Re-download overwrites them with huge base64 TSVs and breaks image_path layout.
    Always skip MD5 checks when this patch is applied (launcher/scoring).
    """
    from vlmeval.dataset.image_base import ImageBaseDataset
    from vlmeval.smp import LMUDataRoot, load, md5  # noqa: F401 — kept for clarity
    import warnings

    if getattr(ImageBaseDataset, "_vtb_md5_patched", False):
        return

    original_prepare = ImageBaseDataset.prepare_tsv

    def prepare_tsv(self, url, file_md5=None):
        """Prefer local TSV; never re-download on MD5 mismatch."""
        import os.path as osp

        from vlmeval.smp import LMUDataRoot, file_size, load, download_file

        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = f"{self.dataset_name}.tsv"
        data_path = osp.join(data_root, file_name)
        self.data_path = data_path

        if osp.exists(data_path):
            # Keep local copy untouched (path-based TSVs won't match official MD5).
            if file_md5 is not None:
                try:
                    from vlmeval.smp import md5 as _md5

                    if _md5(data_path) != file_md5:
                        warnings.warn(
                            f"VTB: MD5 mismatch for {file_name}; keeping local TSV "
                            f"(skip re-download)."
                        )
                except Exception:
                    pass
            if file_size(data_path, "GB") > 1:
                local_path = data_path.replace(".tsv", "_local.tsv")
                if not osp.exists(local_path) or os.environ.get("FORCE_LOCAL", None):
                    from vlmeval.tools import LOCALIZE

                    LOCALIZE(data_path, local_path)
                data_path = local_path
            return load(data_path)

        # Missing local file — fall back to upstream download.
        return original_prepare(self, url, file_md5=file_md5)

    ImageBaseDataset.prepare_tsv = prepare_tsv
    ImageBaseDataset._vtb_md5_patched = True


def patch_lmudata_image_paths() -> None:
    """Map stale absolute / repo-relative image_path entries to LMUData images/."""
    from vlmeval.dataset.image_base import ImageBaseDataset
    from vlmeval.smp import toliststr

    from src.evaluation.lmu_data import parse_image_path_field, resolve_image_path

    if getattr(ImageBaseDataset, "_vtb_image_path_patched", False):
        return

    original_build = ImageBaseDataset.build_prompt
    original_dump = ImageBaseDataset.dump_image

    def _resolve_messages(self, msgs):
        for item in msgs:
            if item.get("type") != "image":
                continue
            paths = toliststr(item["value"])
            resolved = [resolve_image_path(p, self.img_root) for p in paths]
            item["value"] = resolved if len(resolved) > 1 else resolved[0]
        return msgs

    def build_prompt(self, line):
        msgs = original_build(self, line)
        return _resolve_messages(self, msgs)

    def dump_image(self, line):
        if isinstance(line, int):
            row = self.data.iloc[line]
        else:
            row = line
        if "image_path" in getattr(row, "index", row):
            paths = parse_image_path_field(row["image_path"])
            # Relative names (incl. MMMU "['1_1.jpg']") stay relative for decode.
            if paths and any(os.path.isabs(p) or "/" in p or "\\" in p for p in paths):
                fixed = [resolve_image_path(p, self.img_root) for p in paths]
                row = dict(row)
                row["image_path"] = fixed if len(fixed) > 1 else fixed[0]
                line = row
            elif paths:
                row = dict(row)
                row["image_path"] = paths if len(paths) > 1 else paths[0]
                line = row
        paths = original_dump(self, line)
        resolved = [resolve_image_path(p, self.img_root) for p in toliststr(paths)]
        return resolved

    ImageBaseDataset.build_prompt = build_prompt
    ImageBaseDataset.dump_image = dump_image

    # Also wrap subclasses that override build_prompt (e.g. ImageMCQDataset / MMBench).
    seen = {ImageBaseDataset}
    stack = list(ImageBaseDataset.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if "build_prompt" not in cls.__dict__:
            continue
        _orig = cls.build_prompt

        def _make_wrapped(orig):
            def wrapped(self, line, *args, **kwargs):
                msgs = orig(self, line, *args, **kwargs)
                return _resolve_messages(self, msgs)

            return wrapped

        cls.build_prompt = _make_wrapped(_orig)

    ImageBaseDataset._vtb_image_path_patched = True


def patch_mmmu_result_transfer_id(vlmeval_run=None) -> None:
    """Local MMMU TSVs use ``index``; upstream MMMU_result_transfer expects ``id``."""
    import pandas as pd
    from vlmeval.smp import dump, load
    from vlmeval.utils import result_transfer as rt
    from vlmeval.utils.matching_util import can_infer
    import string

    if getattr(rt, "_vtb_mmmu_id_patched", False):
        if vlmeval_run is not None:
            vlmeval_run.MMMU_result_transfer = rt.MMMU_result_transfer
        return

    def MMMU_result_transfer(result_path):
        res = {}
        result_data = load(result_path)
        id_col = "id" if "id" in result_data.columns else "index"
        if id_col not in result_data.columns:
            raise KeyError("MMMU result file missing both 'id' and 'index' columns")
        mcq = result_data["A"].notna() if "A" in result_data.columns else pd.Series([False] * len(result_data))
        for i in range(len(result_data)):
            line = result_data.iloc[i]
            key = line[id_col]
            if mcq.iloc[i]:
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                res[key] = can_infer(line["prediction"], options)
            else:
                res[key] = line["prediction"]
        result_json = result_path.replace(".xlsx", ".json")
        dump(res, result_json)
        return result_json

    rt.MMMU_result_transfer = MMMU_result_transfer
    rt._vtb_mmmu_id_patched = True
    if vlmeval_run is not None:
        vlmeval_run.MMMU_result_transfer = MMMU_result_transfer


def patch_decode_non_rgb_images() -> None:
    """PNG cannot encode CMYK/etc.; convert to RGB before saving decoded images."""
    from vlmeval.smp import vlm as vlm_smp

    if getattr(vlm_smp, "_vtb_rgb_decode_patched", False):
        return

    original = vlm_smp.decode_base64_to_image

    def decode_base64_to_image(base64_string, target_size=-1):
        image = original(base64_string, target_size=target_size)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image

    vlm_smp.decode_base64_to_image = decode_base64_to_image
    vlm_smp._vtb_rgb_decode_patched = True


def apply_vlmeval_scoring_patches() -> None:
    """Patches needed when loading datasets for offline rescore/finalize."""
    os.environ.setdefault("VTB_ALLOW_SUBSAMPLED_TSV", "1")
    patch_subsampled_tsv_md5()
    patch_lmudata_image_paths()
    patch_mmmu_result_transfer_id()
    patch_decode_non_rgb_images()
