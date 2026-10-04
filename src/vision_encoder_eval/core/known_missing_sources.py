"""Missing in the captured original repository, not introduced by migration."""
MISSING_LEGACY_MODULES = {
    'vision_encoder_eval.mllm.utils.wandb_eval': 'Optional logging; historical code catches ImportError.',
    'vision_encoder_eval.mllm.IBQ.util': 'Unreleased upstream LPIPS checkpoint helper; not a supported active training-loss route.',
    'vision_encoder_eval.mllm.IBQ.modules.util': 'Historical IBQ import namespace is absent.',
    'vision_encoder_eval.mllm.IBQ.modules.losses.lpips': 'Historical IBQ import namespace is absent.',
    'vision_encoder_eval.mllm.IBQ.modules.losses.vqperceptual': 'Historical IBQ import namespace is absent.',
    'vision_encoder_eval.mllm.IBQ.modules.discriminator.model': 'Historical IBQ import namespace is absent.',
}
