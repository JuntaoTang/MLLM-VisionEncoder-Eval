"""Explicit entry points; discovery does not import heavy model dependencies."""
from dataclasses import dataclass

@dataclass(frozen=True)
class WorkerSpec:
    module: str
    environment: str
    blocked: str = ''

WORKERS = {}

def _add(group, environment, names):
    for name in names.split():
        WORKERS[f'{group}.{name}'] = WorkerSpec(f'vision_encoder_eval.workers.{group}.{name}', environment)

_add('linear', 'probing', 'linear_probe linear_probe_cached summarize_results')
_add('alignment', 'probing', 'build_data build_cc3m2k encode_text encode_vision train_align summarize summarize_budget robustness budget correlate flops flops_cc3m')
_add('law', 'mllm', 'compute_a_score compute_c_score extract_c_features extract_c_features_discrete compute_a_score_discrete fit_ac eval_gt_ac_policy eval_k8_protocol dump_policy_raw estimate_ac_flops estimate_ac_flops_gt70 inventory')
_add('tokbench', 'tokbench', 'compute_all_metrics eval_text eval_face summarize_paper_results check_eval_requirements')
_add('tokbench.tokenzier_vae_scripts.image_scripts', 'tokbench', 'toklip_l_rec toklip_s_rec unitok_vae_rec vilau_rec resize_rec')
_add('ckax.scripts', 'ckax', 'prepare_images extract_features extract_discrete_features extract_text_features extract_qwen_text_features compute_cka_kernel compute_crossmodal_stats make_cm_final run_ckax run_budget_sweep verify_ckax')
WORKERS['zero_shot.metaclip2'] = WorkerSpec('vision_encoder_eval.workers.zero_shot.models.metaclip2.zero_shot_metaclip2_imagenet', 'zero_shot')
WORKERS['mllm.pipeline'] = WorkerSpec('vision_encoder_eval.workers.mllm_entry', 'mllm')
WORKERS['zero_shot.clip_benchmark'] = WorkerSpec('vision_encoder_eval.workers.zero_clip', 'zero_shot')
_add('knn.different_shot', 'knn', 'evaluate_features run_preexported_features')
_add('knn.protocols', 'knn', 'build_protocol')
_add('knn.different_shot.models', 'knn', 'siglip dinov2 vila_u_256 mae metaclip_worldwide dinov3 toklip rae pe_core eupe metaclip web_dino dino pixio uniar_bsq unitok ijepa')

def get_worker(name):
    try:
        return WORKERS[name]
    except KeyError as exc:
        raise ValueError(f'unknown worker {name!r}; available: {sorted(WORKERS)}') from exc
