"""First-party tower construction reused without altering feature taps."""

def build_vision_tower(vision_config, device='cuda:0'):
    from ..workers.law.common import build_vision_tower as build
    return build(vision_config,device)
