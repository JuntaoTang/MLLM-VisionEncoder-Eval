"""Dependency-light configuration and evidence checks for acceptance scripts."""
import json
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

FLAGS=('VEE_VLMEVAL_PIPELINE','VEE_DISCRETE_TRAINER','VEE_DISCRETE_FACTORY','VEE_CONTINUOUS_TRAINER',
       'VEE_LINEAR_BUDGET','VEE_LINEAR_CACHE','VEE_ZERO_SHOT_PIPELINE',
       'VEE_TOKBENCH_METRICS','VEE_TOKBENCH_RECONSTRUCTION','VEE_LAW_MODEL_PIPELINE')
ASSETS={'knn_exports':'VEE_REAL_KNN_ROOT','continuous_tokenizers':'VEE_REAL_TOKENIZER_ROOT',
        'continuous_checkpoint':'VEE_REAL_MODEL_CHECKPOINT','discrete_part':'VEE_REAL_DISCRETE_PART',
        'discrete_llm_metadata':'VEE_REAL_DISCRETE_LLM_ROOT'}
LAYERS={'continuous_trainer':'VEE_CONTINUOUS_TRAINER_TEST_DEPENDENCIES','vlmeval':'VEE_VLMEVAL_TEST_DEPENDENCIES',
        'probing':'VEE_PROBING_TEST_DEPENDENCIES','zero_shot':'VEE_ZERO_SHOT_TEST_DEPENDENCIES',
        'tokbench':'VEE_TOKBENCH_TEST_DEPENDENCIES'}


def resolve_setup(path):
    path=Path(path).resolve(); config=json.loads(path.read_text())
    if set(config)!={'schema_version','python','output_root','dependencies','assets'} or config['schema_version']!=1:
        raise ValueError('acceptance config must match acceptance.local.example.json')
    def resolved(value):
        if not isinstance(value,str) or not value.strip(): raise ValueError('explicit nonempty path required')
        result=Path(value).expanduser()
        return (path.parent/result).resolve() if not result.is_absolute() else result.resolve()
    python=resolved(config['python'])
    if not python.is_file() or not os.access(python,os.X_OK): raise ValueError('configured Python is not executable')
    if not isinstance(config['assets'],dict) or set(config['assets'])!=set(ASSETS): raise ValueError('all pinned asset routes must be configured')
    assets={key:resolved(value) for key,value in config['assets'].items()}
    for key,value in assets.items():
        if not value.exists(): raise ValueError(f'missing acceptance asset {key}: {value}')
    dependencies=config['dependencies']
    if not isinstance(dependencies,dict) or set(dependencies)!={'base',*LAYERS}: raise ValueError('explicit isolated dependency layers required')
    layers={}
    for key,values in dependencies.items():
        if not isinstance(values,list): raise ValueError('dependency layers must be path lists; use [] for installed venv packages')
        layers[key]=[resolved(value) for value in values]
        if any(not value.is_dir() for value in layers[key]): raise ValueError(f'missing dependency directory: {key}')
    return config,python,assets,layers,resolved(config['output_root'])


def junit_summary(path):
    document=ET.parse(path)
    suites=list(document.getroot().iter('testsuite'))
    counts={key:sum(int(suite.get(key,'0')) for suite in suites) for key in ('tests','failures','errors','skipped')}
    if counts['tests']==0 or any(counts[key] for key in ('failures','errors','skipped')):
        raise RuntimeError(f'selected acceptance requires zero failures/errors/skips: {counts}')
    cases=[(case.get('classname',''),case.get('name','')) for case in document.getroot().iter('testcase')]
    if len(cases)!=counts['tests'] or len(set(cases))!=len(cases) or any(not name for _,name in cases):
        raise RuntimeError('JUnit counts must describe distinct named test cases')
    counts['test_cases_sha256']=hashlib.sha256(json.dumps(sorted(cases),separators=(',',':')).encode()).hexdigest()
    return counts


def validate_matching_suites(source,wheel):
    if not source.get('test_cases_sha256') or source!=wheel:
        raise RuntimeError('Source and clean-wheel acceptance must run the same complete test cases')


def acceptance_snapshot(root):
    """Audit the tested source tree; exclude caches, build metadata and locals.

    No archived source/capture is read. Documentation can be updated while
    tests run, but active code/config/test/upstream changes invalidate the run.
    """
    root=Path(root).resolve()
    paths=[]
    for name in ('src','tests','configs','scripts','requirements','third_party'):
        for path in (root/name).rglob('*'):
            if not path.is_file(): continue
            if any(part in {'__pycache__','.git','.pytest_cache','legacy_capture'} or
                   part.endswith('.egg-info') for part in path.relative_to(root).parts): continue
            if path.name in {'local.yaml','acceptance.local.json'} or path.suffix in {'.pyc','.pyo'}: continue
            paths.append(path)
    paths += [root/name for name in ('pyproject.toml','setup.py','tools/run_acceptance.py','tools/check_clean_wheel.py')
              if (root/name).is_file()]
    digest=hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {'sha256':digest.hexdigest(),'files':len(paths)}
