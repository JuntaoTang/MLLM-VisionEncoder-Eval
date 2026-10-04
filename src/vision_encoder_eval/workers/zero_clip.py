"""Run the established upstream protocol without cloning or installing at runtime."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

def upstream_receipt(source_root, expected_commit):
    """Require the declared clean checkout; never clone/fetch at run time."""
    root=Path(source_root).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}',expected_commit):
        raise ValueError('clip_benchmark requires a full reviewed Git commit, not main/tag')
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
    if Path(git('rev-parse','--show-toplevel')).resolve()!=root or git('rev-parse','HEAD')!=expected_commit:
        raise ValueError('clip_benchmark checkout differs from the configured commit')
    if git('status','--porcelain'):
        raise ValueError('clip_benchmark checkout is dirty')
    module=importlib.import_module('clip_benchmark.cli')
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError('installed clip_benchmark does not load from the declared checkout')
    digest=hashlib.sha256()
    files=sorted(git('ls-files','clip_benchmark').splitlines())
    if not files: raise ValueError('clip_benchmark checkout contains no declared package sources')
    for relative in files:
        path=root/relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('invalid clip_benchmark source file')
        digest.update(relative.encode()); digest.update(b'\0'); digest.update(hashlib.sha256(path.read_bytes()).digest())
    return module,{'upstream_commit':expected_commit,'source_root':str(root),
                   'package_source_sha256':digest.hexdigest(),'files':len(files)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--imagenet-root', required=True)
    p.add_argument('--model-file', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--upstream-source',required=True)
    p.add_argument('--upstream-commit',required=True)
    args = p.parse_args()
    # Same flags as both historical family launchers; upstream version is
    # installed and pinned by the user's isolated zero_shot environment.
    benchmark,receipt=upstream_receipt(args.upstream_source,args.upstream_commit)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir)/'upstream_revision.json').write_text(json.dumps(receipt,indent=2)+'\n')
    sys.argv = ['clip_benchmark', 'eval', '--dataset', 'imagenet1k',
                '--dataset_root', args.imagenet_root, '--task', 'zeroshot_classification',
                '--model_type', 'open_clip', '--pretrained_model', args.model_file,
                '--output', str(Path(args.output_dir)/'{dataset}_{model}_{pretrained}_{language}_{task}.json')]
    status=benchmark.main()
    if status not in (None,0): return status
    # A revision receipt is not a scientific result. An upstream no-op or
    # malformed score must not satisfy the framework's nonempty-dir contract.
    scores=sorted(path for path in Path(args.output_dir).glob('*.json')
                  if path.name!='upstream_revision.json')
    if not scores:
        raise RuntimeError('clip_benchmark produced no scientific result JSON')
    from ..reporting.native import load_native_metrics
    for path in scores: load_native_metrics('clip_benchmark',path)
    return status

if __name__ == '__main__':
    raise SystemExit(main())
