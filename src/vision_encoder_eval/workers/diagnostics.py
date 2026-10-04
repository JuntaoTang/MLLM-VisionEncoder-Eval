"""Read-only, selected-route dependency discovery in the worker interpreter.

Does not run a model, download assets, or import the experiment entry point.
This is a preflight check, not a replacement for a real model smoke test.
"""
import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

def requirements(worker):
    if worker.startswith('knn.'):
        return ('numpy','faiss'), ()
    if worker in {'linear.linear_probe','linear.linear_probe_cached'}:
        return ('numpy','torch','torchvision','fvcore','tqdm'), (
            'third_party/dinov2/dinov2/data/__init__.py',
            'third_party/dinov2/dinov2/data/transforms.py',
            'third_party/dinov2/dinov2/eval/metrics.py',)
    if worker.startswith('mllm.') or worker.startswith('law.') and any(
            term in worker for term in ('a_score','extract_c_features')):
        return ('numpy','torch','transformers','pandas','PIL','yaml'), (
            'third_party/LLaVA-NeXT/llava/model/multimodal_encoder/builder.py',)
    if worker.startswith('alignment.encode'):
        return ('numpy','torch','transformers','PIL','tqdm'), ()
    if worker.startswith('alignment.') and worker.split('.')[-1] in {
            'train_align','robustness','budget','correlate','flops','flops_cc3m'}:
        return ('numpy','torch','scipy','tqdm'), ()
    if worker.startswith('ckax.'):
        return ('numpy','torch','scipy','sklearn'), ()
    if worker=='zero_shot.metaclip2':
        return ('numpy','torch','torchvision','open_clip','transformers','sentencepiece','huggingface_hub'), ()
    if worker=='zero_shot.clip_benchmark':
        return ('clip_benchmark','open_clip'), ()
    if worker=='tokbench.eval_face':
        return ('numpy','PIL','cv2','insightface','onnxruntime'), ()
    if worker=='tokbench.eval_text':
        return ('numpy','PIL','torch','cv2'), ('third_party/doctr/models/__init__.py',)
    if worker.startswith('tokbench.') and 'rec' in worker:
        return ('numpy','torch','torchvision','transformers','PIL'), ()
    return (), ()

def inspect_worker(worker,resource_root):
    from .registry import get_worker
    get_worker(worker)
    modules,resources = requirements(worker)
    missing = {}
    for module in modules:
        try:
            if importlib.util.find_spec(module) is None:
                missing['module:'+module] = 'not installed in the configured interpreter'
        except (ImportError,ValueError,AttributeError) as exc:
            missing['module:'+module] = str(exc)
    for relative in resources:
        path = Path(resource_root)/relative
        if not path.is_file():
            missing['resource:'+relative] = str(path)
    # A module's presence does not prove that its selected scientific API exists.
    # In particular LLaVA's Transformers 4.51.3 lacks MetaClip2 altogether.
    apis = {'transformers': ('MetaClip2Config','MetaClip2Model')} if worker=='zero_shot.metaclip2' else {}
    for module, names in apis.items():
        if 'module:'+module in missing:
            continue
        try:
            imported = importlib.import_module(module)
            for name in names:
                if not hasattr(imported,name):
                    missing['api:'+module+'.'+name] = 'not available in the configured interpreter'
        except Exception as exc:
            missing['api:'+module] = str(exc)
    return {'status':'failed' if missing else 'success','worker':worker,
            'python':sys.executable,'required_modules':list(modules),
            'required_resources':list(resources),'missing':missing,
            'required_apis': {module:list(names) for module,names in apis.items()},
            'scope':'critical dependencies only; real model smoke is still required'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',required=True)
    parser.add_argument('--resource-root',required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_worker(args.worker,args.resource_root),sort_keys=True))

if __name__=='__main__': main()
