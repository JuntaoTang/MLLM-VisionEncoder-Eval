from __future__ import annotations
import math
import torch
import torch.nn.functional as F

# Metadata supplied explicitly by the dataset/encoder adapter.
IMAGENET_CLASSNAMES = ()
OPENAI_IMAGENET_TEMPLATES = ()

def projected(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output

def build_classifier(model, tokenizer, device, text_batch_size, *, class_names=None, templates=None):
    names = IMAGENET_CLASSNAMES if class_names is None else class_names
    prompt_templates = OPENAI_IMAGENET_TEMPLATES if templates is None else templates
    if not names or not prompt_templates:
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES as defaults, OPENAI_IMAGENET_TEMPLATES as default_templates
        names = defaults if class_names is None else class_names
        prompt_templates = default_templates if templates is None else templates
    if text_batch_size <= 0 or not names or not prompt_templates:
        raise ValueError('class names, templates and a positive text batch size are required')
    prompts = [template(name) for name in names for template in prompt_templates]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(prompts), text_batch_size):
            tokens = {k: v.to(device) for k, v in tokenizer(prompts[start : start + text_batch_size]).items()}
            batches.append(F.normalize(projected(model.get_text_features(**tokens)).float(), dim=-1).cpu())
    features = torch.cat(batches).view(len(names), len(prompt_templates), -1)
    return F.normalize(features.mean(1), dim=-1).T.contiguous().to(device=device, dtype=torch.bfloat16)

def auto_batch_size(image_size, vision_width, requested):
    if requested:
        return requested
    if image_size >= 378 and vision_width >= 1280:
        return 96
    if image_size >= 378:
        return 256
    if vision_width >= 1024:
        return 256
    return 512
