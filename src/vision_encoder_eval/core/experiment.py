"""Content-addressed experiment execution, isolated workers and suite reporting."""
from __future__ import annotations

import importlib.resources
import importlib.metadata
import json
import os
import subprocess
import sys
import re
from copy import deepcopy
from itertools import product
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from ..config import ConfigError, resolve_config
from ..workers.registry import get_worker
from .artifacts import FileRecord, read_json, write_json_atomic, validate_result_payload
from .hashing import sha256_file, sha256_json
from .registry import METHODS


def matrix_configs(resolved):
    from ..config.loader import ResolvedConfig
    axes = resolved.value['experiment'].get('matrix')
    if axes is None:
        return None
    if not isinstance(axes,dict) or not axes:
        raise ConfigError('matrix must be a non-empty axis mapping')
    for name,values in axes.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) or not isinstance(values,list) or not values:
            raise ConfigError('matrix axes require identifier names and non-empty value lists')
        if not all(isinstance(v,(str,int,float,bool)) for v in values) or len({sha256_json(v) for v in values})!=len(values):
            raise ConfigError('matrix axis values must be unique JSON scalars')
    count = 1
    for values in axes.values():
        count *= len(values)
    if count > 10000:
        raise ConfigError('matrix exceeds 10000 runs; split the suite explicitly')
    children = []
    for combination in product(*axes.values()):
        values = dict(zip(axes,combination))
        def substitute(item):
            if isinstance(item,str):
                for name,value in values.items():
                    if item == '${matrix.'+name+'}':
                        return value
                    item = item.replace('${matrix.'+name+'}',str(value))
                if '${matrix.' in item:
                    raise ConfigError(f'unresolved matrix token: {item}')
                return item
            if isinstance(item,list):
                return [substitute(v) for v in item]
            if isinstance(item,dict):
                return {k:substitute(v) for k,v in item.items()}
            return item
        value = deepcopy(dict(resolved.value))
        del value['experiment']['matrix']
        value['experiment'] = substitute(value['experiment'])
        label = re.sub('[^A-Za-z0-9_.-]+','-', '-'.join(str(v) for v in values.values()))[:100]
        value['experiment']['experiment']['name'] += '-'+label+'-'+sha256_json(values)[:8]
        value['experiment']['matrix_values'] = values
        children.append(ResolvedConfig(value=value,sha256=sha256_json(value)))
    return children


def input_records(experiment):
    records = {}
    for name, raw in experiment.get('inputs', {}).items():
        if not isinstance(raw, str):
            raise ConfigError(f'inputs.{name} must be a file path')
        path = Path(raw)
        if path.is_file():
            records[name] = {'path': str(path), 'sha256': sha256_file(path), 'kind':'file'}
        elif path.is_dir():
            entries = {p.relative_to(path).as_posix():sha256_file(p) for p in sorted(path.rglob('*')) if p.is_file()}
            if not entries:
                raise ConfigError(f'input directory is empty: {path}')
            records[name] = {'path':str(path),'sha256':sha256_json(entries),'kind':'directory','files':len(entries)}
        else:
            raise ConfigError(f'input missing: {path}')
    return records


def source_hash(resource_root=None):
    package = Path(__file__).resolve().parents[1]
    records = {p.relative_to(package).as_posix(): sha256_file(p)
               for p in sorted(package.rglob('*')) if p.is_file()
               and '__pycache__' not in p.parts and p.suffix in {'.py', '.yaml', '.json', '.tsv'}}
    if resource_root is not None:
        upstream = Path(resource_root)/'third_party'
        for p in sorted(upstream.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
                records['third_party/'+p.relative_to(upstream).as_posix()] = sha256_file(p)
    return sha256_json(records)


def environment_records(resolved):
    names = ['numpy','torch','torchvision','transformers','scipy','scikit-learn','faiss-cpu',
             'PyYAML','timm','open_clip_torch','clip-benchmark','deepspeed','accelerate']
    core = {}
    for name in names:
        try:
            core[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {'core':{'python':str(Path(sys.executable).resolve()),'version':sys.version,'packages':core}}
    interpreters = set()
    local = resolved.value['local']
    for step in resolved.value['experiment'].get('steps', []):
        spec = get_worker(step['worker'])
        interpreters.add(local['environments'][step.get('environment',spec.environment)]['python'])
    if not resolved.value['experiment'].get('steps') and resolved.value['experiment']['experiment']['kind']=='method':
        python = method_python(resolved)
        if Path(python).resolve()!=Path(sys.executable).resolve():
            interpreters.add(python)
    for python in sorted(interpreters):
        code = (
            'import sys,json,importlib.metadata as m; '
            f'names={names!r}; installed={{d.metadata["Name"].lower():d.version for d in m.distributions()}}; '
            'print(json.dumps({"python":sys.executable,"version":sys.version,"packages":'
            '{n:installed[n.lower()] for n in names if n.lower() in installed}}))'
        )
        completed = subprocess.run([python,'-c',code],capture_output=True,text=True,check=False,timeout=30)
        if completed.returncode:
            raise ConfigError(f'cannot inspect worker environment {python}: {completed.stderr}')
        result[python] = json.loads(completed.stdout)
    return result


def method_python(resolved):
    experiment = resolved.value['experiment']
    name = experiment.get('runtime',{}).get('environment','core')
    config = resolved.value['local'].get('environments',{}).get(name)
    if config is None and name=='core':
        return sys.executable
    if not config or not config.get('python'):
        raise ConfigError(f'configure local.environments.{name}.python')
    return config['python']


def _execute_method(resolved,directory):
    python = method_python(resolved)
    runtime = {**resolved.value['local'].get('runtime',{}),**resolved.value['experiment'].get('runtime',{})}
    if Path(python).resolve()==Path(sys.executable).resolve() and 'cuda_visible_devices' not in runtime:
        return dict(METHODS.get(resolved.value['experiment']['method']).runner(
            {**resolved.value,'config_sha256':resolved.sha256}))
    request = directory/'numeric_request.json'
    response = directory/'numeric_result.json'
    resolved.write(request)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    if 'cuda_visible_devices' in runtime:
        env['CUDA_VISIBLE_DEVICES'] = str(runtime['cuda_visible_devices'])
    with (directory/'numeric.log').open('w') as log:
        completed = subprocess.run([python,'-m','vision_encoder_eval.workers.numeric',
                                   '--request',str(request),'--result',str(response)],
                                  cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    if completed.returncode or not response.is_file():
        raise ConfigError(f'numerical worker failed; inspect {directory/"numeric.log"}')
    return read_json(response)


def plan(resolved):
    experiment = resolved.value['experiment']
    kind = experiment['experiment']['kind']
    if kind == 'suite':
        reporting=experiment.get('reporting')
        if reporting is not None and reporting!={'format':'long_table'}:
            from ..reporting.panel import validate_panel_spec
            try:
                validate_panel_spec(reporting)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        return {'kind': kind, 'experiments': experiment['experiments']}
    if kind == 'report':
        return {'kind': kind}
    if 'report_cell' in experiment:
        from ..reporting.panel import validate_cell
        validate_cell(experiment['report_cell'])
    steps = experiment.get('steps')
    if steps is not None:
        if not isinstance(steps, list) or not steps:
            raise ConfigError('steps must be a non-empty list')
        local = resolved.value['local']
        for step in steps:
            spec = get_worker(step.get('worker'))
            if spec.blocked:
                raise ConfigError(spec.blocked)
            argv = step.get('arguments', [])
            if not isinstance(argv, list) or not all(isinstance(v, (str, int, float)) for v in argv):
                raise ConfigError('worker arguments must be a flat argv list')
            environment = step.get('environment', spec.environment)
            if not local.get('environments', {}).get(environment, {}).get('python'):
                raise ConfigError(f'configure local.environments.{environment}.python')
            if not isinstance(step.get('outputs'), list) or not step['outputs']:
                raise ConfigError('each worker step must declare non-empty outputs; exit zero alone is not proof of an experiment result')
            if not experiment.get('inputs'):
                raise ConfigError('worker experiments require pinned input files/manifests under inputs')
            if step.get('metrics_adapter') is not None:
                from ..reporting.native import FORMATS
                adapter = step['metrics_adapter']
                if not isinstance(adapter,dict) or adapter.get('format') not in FORMATS:
                    raise ConfigError('invalid explicit native metrics adapter')
                if adapter.get('path') not in step['outputs']:
                    raise ConfigError('native metrics path must be a declared step output')
                if set(adapter) - {'format', 'path', 'expected_models'}:
                    raise ConfigError('unknown native metrics adapter field')
                names = adapter.get('expected_models')
                if names is not None and (not isinstance(names, list) or not names or
                    not all(isinstance(name, str) and name.strip() for name in names) or len(set(names)) != len(names)):
                    raise ConfigError('native expected_models must be a nonempty ordered unique name list')
            if spec.module.endswith('mllm_entry') and '--mode' in argv:
                if argv.index('--mode')+1 >= len(argv):
                    raise ConfigError('--mode requires a value')
        return {'kind': kind, 'steps': steps}
    if kind != 'method':
        raise ConfigError(f'{kind} requires explicit worker steps')
    spec = METHODS.get(experiment['method'])
    if spec.execution_mode=='worker':
        raise ConfigError(f"{experiment['method']} requires configured worker steps")
    required = {'database','database_labels','query'} if experiment['method']=='knn' else {'visual_features','text_features'}
    if not required.issubset(experiment.get('inputs',{})):
        raise ConfigError(f'method inputs missing: {sorted(required-set(experiment.get("inputs",{})))}')
    if experiment['method']!='knn' and not {'visual_manifest','text_manifest'}.issubset(experiment['inputs']) and experiment.get('protocol',{}).get('assume_aligned_rows') is not True:
        raise ConfigError('paired features require visual/text row manifests or explicit assume_aligned_rows=true')
    return {'kind': kind, 'method': experiment['method']}


def preflight(resolved, ancestors=()):
    expanded = matrix_configs(resolved)
    if expanded is not None:
        records = [preflight(child,ancestors) for child in expanded]
        return {'status':'success' if all(v['status']=='success' for v in records) else 'failed',
                'matrix':records,'config_sha256':resolved.sha256}
    execution = plan(resolved)
    experiment = resolved.value['experiment']
    local = resolved.value['local']
    source = resolved.value['sources']['experiment']
    if source in ancestors:
        raise ConfigError(f'cyclic suite: {source}')
    if execution['kind'] == 'suite':
        children = {child:preflight(resolve_config(local_path=resolved.value['sources']['local'], experiment_path=child),
                                    (*ancestors,source)) for child in experiment['experiments']}
        return {'status':'success' if all(v['status']=='success' for v in children.values()) else 'failed',
                'experiments':children, 'config_sha256':resolved.sha256}
    missing = {}
    environment_checks = {}
    for name in experiment.get('required_paths', []):
        path = local['paths'].get(name)
        if path is None or not Path(path).exists():
            missing[name] = path
    for step in experiment.get('steps', []):
        spec = get_worker(step['worker'])
        environment = step.get('environment', spec.environment)
        python = local['environments'][environment]['python']
        if not Path(python).is_file():
            missing[f'interpreter:{environment}'] = python
        else:
            from .runtime import repository_root
            resources = Path(local.get('resource_root') or repository_root())
            env = dict(os.environ)
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
            try:
                completed = subprocess.run([python,'-m','vision_encoder_eval.workers.diagnostics',
                    '--worker',step['worker'],'--resource-root',str(resources)],env=env,
                    capture_output=True,text=True,check=False,timeout=30)
                if completed.returncode:
                    raise ConfigError(completed.stderr.strip() or 'dependency discovery failed')
                check = json.loads(completed.stdout)
                environment_checks[step['worker']] = check
                if check['status']!='success':
                    missing['worker:'+step['worker']] = check['missing']
            except (subprocess.TimeoutExpired,ConfigError,ValueError,OSError) as exc:
                missing['worker:'+step['worker']] = str(exc)
    if execution.get('method'):
        python = method_python(resolved)
        if not Path(python).is_file():
            missing['interpreter:method'] = python
    try:
        records = input_records(experiment)
    except ConfigError as exc:
        missing['inputs'] = str(exc)
        records = {}
    return {'status': 'failed' if missing else 'success', 'missing_paths': missing,
            'inputs': records, 'execution': execution, 'config_sha256': resolved.sha256,
            'environment_checks':environment_checks}


def _render_configs(workspace, local):
    import yaml
    configs = importlib.resources.files('vision_encoder_eval')/'resources'/'mllm_configs'
    prefixes = {
        '/cache/ckpt/download': 'download', '/cache/ckpt/trained': 'trained',
        '/cache/data': 'datasets', '/cache/VTB': 'runtime',
        '/cache/wangky/ocr_exp/VTB': 'mllm',
        '/home/ma-user/work_space/VTB': 'mllm',
    }
    paths = {**local['paths'], 'mllm': str(workspace)}
    prefixes.update(local.get('path_rewrites', {}))
    def render(value):
        if isinstance(value, str):
            for prefix, key in sorted(prefixes.items(), key=lambda item: -len(item[0])):
                if value == prefix or value.startswith(prefix+'/'):
                    if key not in paths:
                        raise ConfigError(f'config asset requires local.paths.{key}')
                    return str(Path(paths[key])/value[len(prefix):].lstrip('/'))
            return value
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {k: render(v) for k,v in value.items()}
        return value
    for source in Path(str(configs)).rglob('*.yaml'):
        destination = workspace/'configs'/source.relative_to(Path(str(configs)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(render(yaml.safe_load(source.read_text())), sort_keys=False))


def _execute_workers(resolved, run_directory):
    from .runtime import repository_root
    local = resolved.value['local']
    experiment = resolved.value['experiment']
    workspace = run_directory/'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    paths = {**local['paths'], 'runtime': str(workspace)}
    environment['VEE_PATHS'] = json.dumps(paths)
    environment['VEE_MLLM_ROOT'] = str(workspace)
    resources = Path(local.get('resource_root') or repository_root())
    environment['VEE_RESOURCE_ROOT'] = str(resources)
    package_parent = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join([package_parent,str(resources/'third_party'),environment.get('PYTHONPATH','')])
    runtime = {**local.get('runtime', {}), **experiment.get('runtime', {})}
    if 'cuda_visible_devices' in runtime:
        environment['CUDA_VISIBLE_DEVICES'] = str(runtime['cuda_visible_devices'])
    worker_env = {
        'ALIGN_DATA_DIR': workspace/'alignment/data', 'ALIGN_CACHE_DIR': workspace/'alignment/cache',
        'ALIGN_RESULTS_DIR': workspace/'alignment/results', 'VEE_LAW_RESULTS': workspace/'law/results',
        'VEE_LAW_LOGS': workspace/'law/logs', 'VEE_CKAX_RESULTS': workspace/'ckax/results',
        'LMU_DATA': paths.get('lmudata', ''), 'CKA_X_GT': paths.get('ground_truth',''),
        'VTB_ROOT': workspace,
        'KNN_PROTOCOL': experiment.get('inputs',{}).get('protocol',''),
        'VTB_CONFIGS_ROOT': workspace/'configs',
        'VEE_CKAX_WORKSPACE': workspace/'ckax',
    }
    environment.update({key: str(value) for key,value in worker_env.items() if value})
    # Cached alignment training/retrieval and data builders never load MLLM
    # recipes. Do not require unrelated base-model/data paths for these routes.
    if any(s['worker'].startswith(('mllm.', 'law.')) or s['worker'] in {
            'alignment.encode_vision', 'alignment.flops', 'alignment.flops_cc3m'}
            for s in experiment['steps']):
        _render_configs(workspace, {**local, 'paths':paths})
    if experiment.get('inputs',{}).get('finished_models'):
        write_json_atomic(workspace/'results/finish.json', read_json(experiment['inputs']['finished_models']))
    if (resources/'third_party').is_dir():
        target = workspace/'third_party'
        if not target.exists():
            target.symlink_to(resources/'third_party', target_is_directory=True)
    stub = importlib.resources.files('vision_encoder_eval')/'resources/cuda_stub/bin/nvcc'
    if Path(str(stub)).is_file():
        import shutil
        destination = workspace/'scripts/cuda_stub/bin/nvcc'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(str(stub),destination)
        destination.chmod(0o755)
    events = []
    output_paths = {}
    metrics = {}
    for index, step in enumerate(experiment['steps']):
        spec = get_worker(step['worker'])
        python = local['environments'][step.get('environment', spec.environment)]['python']
        argv = [str(v).replace('${run_dir}', str(run_directory)).replace('${workspace}', str(workspace))
                for v in step.get('arguments', [])]
        request_path = run_directory/f'worker_{index:02d}_request.json'
        response_path = run_directory/f'worker_{index:02d}_result.json'
        write_json_atomic(request_path,{'schema_version':1,'worker':step['worker'],
                                        'arguments':argv,'protocol':experiment.get('protocol',{})})
        command = [python, '-m', 'vision_encoder_eval.workers.dispatch',
                   '--request',str(request_path),'--result',str(response_path)]
        step_env = {**environment, **{k:str(v).replace('${workspace}',str(workspace)).replace('${run_dir}',str(run_directory))
                                     for k,v in step.get('env', {}).items()}}
        log = run_directory/f'step_{index:02d}.log'
        with log.open('w') as handle:
            completed = subprocess.run(command, cwd=workspace, env=step_env, stdout=handle,
                                       stderr=subprocess.STDOUT, check=False)
        events.append({'worker':step['worker'], 'command':command, 'returncode':completed.returncode,
                       'log': log.name})
        write_json_atomic(run_directory/'events.json', events)
        if completed.returncode:
            raise ConfigError(f"worker {step['worker']} failed (exit {completed.returncode}); inspect {log}")
        if not response_path.is_file():
            raise ConfigError(f'worker did not write a JSON result: {response_path}')
        response = read_json(response_path)
        if response.get('status')!='success' or response.get('worker')!=step['worker'] or response.get('returncode')!=0:
            raise ConfigError(f'invalid worker response: {response_path}')
        for raw in step.get('outputs', []):
            output = Path(str(raw).replace('${workspace}', str(workspace)).replace('${run_dir}', str(run_directory)))
            if not output.is_absolute():
                output = workspace/output
            if not output.exists():
                raise ConfigError(f'worker returned zero but declared output is absent: {output}')
            if output.is_file() and output.stat().st_size == 0:
                raise ConfigError(f'worker returned zero but declared output is empty: {output}')
            if output.is_dir() and not any(p.is_file() and p.stat().st_size > 0 for p in output.rglob('*')):
                raise ConfigError(f'worker returned zero but declared output is empty: {output}')
            output_paths[f'{index}:{raw}'] = str(output)
        if step.get('metrics_adapter'):
            from ..reporting.native import load_native_metrics
            adapter = step['metrics_adapter']
            path = Path(str(adapter['path']).replace('${workspace}',str(workspace)).replace('${run_dir}',str(run_directory)))
            if not path.is_absolute():
                path = workspace/path
            parsed = load_native_metrics(adapter['format'],path,
                                          expected_shots=experiment.get('protocol',{}).get('train_shots'),
                                          expected_models=adapter.get('expected_models'))
            if metrics.keys() & parsed.keys():
                raise ConfigError('native metrics collide across steps')
            metrics.update(parsed)
    return events, input_records({'inputs':output_paths}), metrics


@contextmanager
def _run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'.running.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ConfigError(f'run already locked: {lock}; inspect process before removing a stale lock') from exc
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump({'pid':os.getpid(),'python':sys.executable},handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_experiment(resolved, *, dry_run=False, ancestors=()):
    expanded = matrix_configs(resolved)
    if expanded is not None:
        # Validate every combination before starting any expensive worker.
        for child in expanded:
            plan(child)
        if not dry_run:
            checks = preflight(resolved, ancestors)
            if checks['status'] != 'success':
                raise ConfigError(f'matrix preflight failed: {checks}')
        results = [run_experiment(child,dry_run=dry_run,ancestors=ancestors) for child in expanded]
        payload = {'status':'dry_run' if dry_run else 'success','matrix_runs':results}
        if not dry_run:
            name = resolved.value['experiment']['experiment']['name']
            directory = Path(resolved.value['local']['output_root'])/f'{name}-{resolved.sha256[:12]}'
            write_json_atomic(directory/'matrix.json',payload)
        return payload
    experiment = resolved.value['experiment']['experiment']
    if dry_run or experiment['kind']=='suite':
        return _run_experiment(resolved,dry_run=dry_run,ancestors=ancestors)
    directory = Path(resolved.value['local']['output_root'])/f"{experiment['name']}-{resolved.sha256[:12]}"
    with _run_lock(directory):
        return _run_experiment(resolved,dry_run=dry_run,ancestors=ancestors)


def _run_experiment(resolved, *, dry_run=False, ancestors=()):
    execution = plan(resolved)
    experiment = resolved.value['experiment']
    source = resolved.value['sources']['experiment']
    if source in ancestors:
        raise ConfigError(f'cyclic suite: {source}')
    if execution['kind'] == 'suite':
        if not dry_run:
            checks = preflight(resolved, ancestors)
            if checks['status'] != 'success':
                raise ConfigError(f'suite preflight failed: {checks}')
        results, outcomes = [], []
        for index, child in enumerate(experiment['experiments']):
            child_config = resolve_config(local_path=resolved.value['sources']['local'], experiment_path=child)
            child_name = child_config.value['experiment']['experiment']['name']
            try:
                child_result = run_experiment(child_config, dry_run=dry_run, ancestors=(*ancestors, source))
            except Exception as exc:
                if not dry_run:
                    outcomes.append({'experiment': child_name, 'status': 'failed', 'reason': str(exc),
                                     'config_sha256': child_config.sha256})
                    for pending in experiment['experiments'][index+1:]:
                        pending_config = resolve_config(local_path=resolved.value['sources']['local'], experiment_path=pending)
                        outcomes.append({'experiment': pending_config.value['experiment']['experiment']['name'],
                            'status': 'not_run', 'reason': 'not started after suite predecessor failed',
                            'config_sha256': pending_config.sha256})
                    destination = Path(resolved.value['local']['output_root'])/f"{experiment['experiment']['name']}-{resolved.sha256[:12]}"
                    from ..reporting.aggregate import write_execution_audit
                    audit = write_execution_audit(outcomes, destination/'report', config_sha256=resolved.sha256)
                    write_json_atomic(destination/'suite_failure.json', {'status': 'failed', 'error': str(exc), **audit})
                raise
            results.append(child_result)
            outcomes.append({'experiment': child_name, 'status': 'success',
                             'config_sha256': child_config.sha256,
                             'run_directory': child_result.get('run_directory', '')})
        payload = {'status':'dry_run' if dry_run else 'success', 'experiments':results}
        if not dry_run:
            destination = Path(resolved.value['local']['output_root'])/f"{experiment['experiment']['name']}-{resolved.sha256[:12]}"
            if experiment.get('reporting'):
                from ..reporting import write_report
                def leaves(records):
                    for record in records:
                        if 'experiments' in record:
                            yield from leaves(record['experiments'])
                        elif 'matrix_runs' in record:
                            yield from leaves(record['matrix_runs'])
                        elif record.get('method')!='report':
                            yield record
                report_directory = destination/'report'
                leaf_results=list(leaves(results))
                if experiment['reporting']['format']=='panel_table':
                    from ..reporting.panel import write_panel_report
                    write_panel_report(leaf_results,report_directory,experiment['reporting'])
                else:
                    write_report(leaf_results,report_directory)
                payload['report_directory'] = str(report_directory)
            write_json_atomic(destination/'suite.json', payload)
        return payload
    if dry_run:
        return {'status':'dry_run', 'config_sha256':resolved.sha256, 'execution':execution}
    checks = preflight(resolved)
    if checks['status'] != 'success':
        raise ConfigError(f"preflight failed: {checks['missing_paths']}")
    resource_root = None
    if execution.get('steps'):
        from .runtime import repository_root
        resource_root = resolved.value['local'].get('resource_root') or repository_root()
    provenance = {'config_sha256':resolved.sha256, 'inputs':checks['inputs'],
                  'code_sha256':source_hash(resource_root), 'environments':environment_records(resolved),
                  'execution_environment':{name:os.environ[name] for name in (
                      'CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED',
                      'OMP_NUM_THREADS','MKL_NUM_THREADS') if name in os.environ}}
    if 'report_cell' in experiment:
        provenance['report_cell']=experiment['report_cell']
    fingerprint = sha256_json(provenance)
    run_id = f"{experiment['experiment']['name']}-{resolved.sha256[:12]}"
    directory = Path(resolved.value['local']['output_root'])/run_id
    result_path = directory/'result.json'
    if result_path.exists():
        previous = read_json(result_path)
        validate_result_payload(previous)
        if previous['status'] != 'success' or previous['audit'].get('fingerprint') != fingerprint:
            raise ConfigError(f'existing run differs or failed; use a new experiment name/output_root: {directory}')
        for record in previous['audit'].get('artifacts', []):
            FileRecord(**record).verify(root=directory)
        declared = previous['audit'].get('outputs', {})
        if declared and input_records({'inputs':{name:record['path'] for name,record in declared.items()}}) != declared:
            raise ConfigError('declared worker outputs changed; cannot reuse result')
        return {**previous, 'reused':True, 'run_directory':str(directory)}
    provenance_path = directory/'provenance.json'
    if provenance_path.exists() and read_json(provenance_path) != provenance:
        raise ConfigError(f'partial run has different inputs/code/environment: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    resolved.write(directory/'resolved_config.json')
    write_json_atomic(directory/'provenance.json', provenance)
    try:
        if execution.get('steps'):
            events, outputs, metrics = _execute_workers(resolved, directory)
            result = {'schema_version':1, 'run_id':run_id, 'status':'success',
                      'method':experiment.get('method', experiment['experiment']['kind']),
                      'dataset':experiment.get('dataset',''), 'protocol':experiment.get('protocol', {}),
                      'metrics':metrics, 'audit':{'outputs':outputs}, 'steps':events}
        elif execution['kind'] == 'report':
            results = [read_json(path) for path in experiment.get('inputs',{}).values()]
            if any(r.get('status') != 'success' for r in results):
                raise ConfigError('report contains unsuccessful results')
            from ..reporting import write_report
            write_report(results,directory)
            result = {'schema_version':1, 'run_id':run_id, 'status':'success', 'method':'report',
                      'dataset':'aggregate', 'protocol':{}, 'metrics':{}, 'audit':{}}
        else:
            result = _execute_method(resolved,directory)
        validate_result_payload(result)
        if result['run_id'] != run_id:
            raise ConfigError('runner returned an inconsistent run_id')
        artifacts = [asdict(FileRecord.capture(p, relative_to=directory)) for p in sorted(directory.rglob('*'))
                     if p.is_file() and p.name != '.running.lock' and not p.is_symlink()
                     and not any(parent.is_symlink() for parent in p.parents)]
        result['audit'] = {**result['audit'], **provenance, 'fingerprint':fingerprint, 'artifacts':artifacts}
        write_json_atomic(result_path, result)
    except Exception as exc:
        write_json_atomic(directory/'failure.json', {'status':'failed','error':str(exc),'provenance':provenance})
        raise
    return {**result,'reused':False,'run_directory':str(directory)}
