import csv
import math
from pathlib import Path

from ..core.artifacts import validate_result_payload, write_json_atomic

def write_execution_audit(outcomes, directory, *, config_sha256):
    """Report fail-fast suite execution without admitting failed metric cells."""
    allowed = {'success', 'failed', 'not_run'}
    if not outcomes or any(row.get('status') not in allowed for row in outcomes):
        raise ValueError('execution audit needs explicit success/failed/not_run outcomes')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'scope': 'Suite execution audit, not a verified metric table',
               'config_sha256': config_sha256, 'outcomes': outcomes}
    write_json_atomic(directory/'execution_audit.json', payload)
    counts = {status: sum(row['status'] == status for row in outcomes) for status in allowed}
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# Reproducibility report', '',
             'Scope: failed suite execution audit. This is not a verified metric table or an acceptance declaration.', '',
             f"Success: {counts['success']}; failed: {counts['failed']}; not run: {counts['not_run']}.", '',
             '| Experiment | Status | Reason | Config SHA-256 | Run directory |',
             '|---|---|---|---|---|']
    for row in outcomes:
        lines.append('| ' + ' | '.join(cell(row.get(key, '')) for key in
            ('experiment', 'status', 'reason', 'config_sha256', 'run_directory')) + ' |')
    lines.extend(['', 'No missing or failed score is imputed as zero. Successful runs retain their own result/audit artifacts.',
                  'The original failure is re-raised and the command exits unsuccessfully. Later jobs were not started.', ''])
    (directory/'REPRODUCIBILITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload

def write_report(results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    runs,metrics,provenance = [],[],[]
    seen = set()
    for result in results:
        validate_result_payload(result)
        if result['status']!='success':
            raise ValueError('cannot include an unsuccessful run in a verified report')
        if result['run_id'] in seen:
            raise ValueError(f"duplicate report run: {result['run_id']}")
        seen.add(result['run_id'])
        base = {'run_id':result['run_id'],'method':result['method'],'dataset':result['dataset']}
        runs.append({**base,'status':result['status'],'metrics_available':bool(result['metrics'])})
        provenance.append({**base, 'protocol':result['protocol'], 'audit':result['audit']})
        for name,value in sorted(result['metrics'].items()):
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if not math.isfinite(value):
                    raise ValueError(f'nonfinite report metric: {name}')
                metrics.append({**base,'metric':name,'value':value})
    for name,rows,fields in [
        ('runs.csv',runs,['run_id','method','dataset','status','metrics_available']),
        ('metrics.csv',metrics,['run_id','method','dataset','metric','value'])]:
        with (directory/name).open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    payload = {'schema_version':1,'runs':runs,'metrics':metrics,'provenance':provenance,
               'note':'Worker-native outputs remain under each run. Unparsed/missing metrics are not imputed or averaged.'}
    write_json_atomic(directory/'report.json',payload)
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# Reproducibility report', '',
             'Scope: the verified runs listed below. This report is not a declaration that all refactor acceptance gates or full-paper reproduction passed.', '',
             '| Run | Method | Dataset | Execution | Numeric metrics | Config SHA-256 | Code SHA-256 |',
             '|---|---|---|---|---|---|---|']
    for run, record in zip(runs, provenance):
        audit = record['audit']
        lines.append('| ' + ' | '.join(cell(value) for value in (
            run['run_id'], run['method'], run['dataset'], run['status'],
            'available' if run['metrics_available'] else 'not parsed / unavailable',
            audit.get('config_sha256', 'not recorded'), audit.get('code_sha256', 'not recorded'))) + ' |')
    lines.extend(['', 'Failed/skipped runs are not included in verified tables; an unsuccessful run is rejected, not converted to a zero score.', '',
                  'Input/weight hashes, protocol, interpreter/environment records and native-output hashes are retained per run in `report.json` under `provenance`.', '',
                  'Full-data reproduction (G5): not established by this report.', ''])
    (directory/'REPRODUCIBILITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload
