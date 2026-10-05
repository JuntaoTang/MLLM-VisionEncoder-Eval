"""Explicit native metric adapters. Never guess or fill missing values."""
import math
import re
import csv
from pathlib import Path

from ..core.artifacts import read_json

FORMATS = {'knn_multishot', 'linear_probe', 'linear_probe_fraction', 'alignment_probe', 'ckax', 'ckax_budget',
           'metaclip2_zero_shot', 'clip_benchmark', 'tokbench_summary', 'law_a', 'law_c', 'mllm_finish'}


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{name} outside declared native range')
    return float(value)


def _mapping(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{name} requires a non-empty mapping')
    return value


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('native rows require an explicit nonempty model/cell name')
    return value


def _check_names(actual, expected):
    if len(actual) != len(set(actual)):
        raise ValueError('duplicate native model/cell')
    if expected is not None and actual != list(expected):
        raise ValueError('native models/cells differ from the declared panel order')


def load_native_metrics(format_name, path, **kwargs):
    """Read the explicitly selected file schema, not a guessed extension/schema."""
    if format_name == 'tokbench_summary':
        with Path(path).open(newline='', encoding='utf-8') as handle:
            payload = list(csv.DictReader(handle))
    else:
        payload = read_json(path)
    return read_native_metrics(format_name, payload, **kwargs)


def _other_metrics(format_name, payload, expected_models):
    metrics = {}
    if expected_models is not None and format_name in {'linear_probe', 'linear_probe_fraction', 'clip_benchmark'}:
        raise ValueError('this single-result format does not declare model identities')
    if format_name == 'mllm_finish':
        from ..mllm.utils.finish_tracker import FINISH_SCORE_ORDER
        rows = _mapping(payload, format_name)
        _check_names(list(rows), expected_models)
        required = set(FINISH_SCORE_ORDER) | {'average'}
        for model, row in rows.items():
            _identifier(model); _mapping(row, model)
            scores = _mapping(row['scores'], 'scores')
            if set(scores) != required:
                raise ValueError('MLLM finish scores must contain all eleven benchmarks and native average')
            # Preserve the old stored display scores/rounding and old average.
            # They mix benchmark-native scales; never relabel them accuracy
            # percent, normalize by value, or silently recompute the mean.
            for dataset in (*FINISH_SCORE_ORDER, 'average'):
                metrics[f'{model}/legacy_display_score/{dataset}'] = _number(scores[dataset], dataset)
        return metrics
    if format_name in {'linear_probe', 'linear_probe_fraction'}:
        best = _mapping(_mapping(payload, format_name)['best_classifier'], 'best_classifier')
        # The paper five-shot worker multiplies by 100 in _evaluate; the
        # historical cached/full protocol serializes TorchMetrics fractions.
        # These are distinct explicit formats, never inferred from magnitude.
        unit = 'percent' if format_name == 'linear_probe' else 'fraction'
        upper = 100 if unit == 'percent' else 1
        metrics[f'top1_{unit}'] = _number(best['accuracy'], 'accuracy', 0, upper)
        metrics[f'top5_{unit}'] = _number(best['top5_accuracy'], 'top5_accuracy', 0, upper)
        if metrics[f'top5_{unit}'] < metrics[f'top1_{unit}']:
            raise ValueError('top5 cannot be below top1')
        for key in ('base_lr', 'effective_lr'):
            metrics[key] = _number(best[key], key, 0)
        return metrics
    if format_name == 'alignment_probe':
        _mapping(payload, format_name)
        if expected_models is not None:
            _check_names([_identifier(payload['slug'])], expected_models)
        test = _mapping(payload['test'], 'test')
        for cap in ('1cap', '5cap'):
            for direction in ('i2t', 't2i'):
                group = _mapping(test[f'{direction}_{cap}'], 'retrieval')
                for k in (1, 5, 10):
                    metrics[f'{direction}/R@{k}_percent/{cap}'] = _number(group[f'R@{k}'], 'recall', 0, 100)
            mean = _number(test[f'mean_recall_{cap}'], 'mean recall', 0, 100)
            if mean != _number(payload['score' if cap == '1cap' else 'score_5cap'], 'score', 0, 100):
                raise ValueError('alignment score differs from native mean recall')
            metrics[f'mean_recall_percent/{cap}'] = mean
        return metrics
    if format_name == 'clip_benchmark':
        native = _mapping(_mapping(payload, format_name)['metrics'], 'metrics')
        for key in ('acc1', 'acc5'):
            metrics[f'{key}_fraction'] = _number(native[key], key, 0, 1)
        if metrics['acc5_fraction'] < metrics['acc1_fraction']:
            raise ValueError('top5 cannot be below top1')
        return metrics
    if format_name == 'ckax':
        rows = _mapping(_mapping(payload, format_name)['results'], 'results')
        _check_names(list(rows), expected_models)
        for cell, row in rows.items():
            _identifier(cell); _mapping(row, cell)
            for metric in ('rho', 'pearson', 'top1_gt'):
                pair = row[f'fold_mean_{metric}']
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('CKA-X fold_mean requires [mean, std]')
                # top1_gt is the selected encoder's GT score, NOT top1 accuracy.
                bounds = (None, None) if metric == 'top1_gt' else (-1, 1)
                metrics[f'{cell}/fold_mean/{metric}'] = _number(pair[0], metric, *bounds)
                metrics[f'{cell}/fold_std/{metric}'] = _number(pair[1], metric + ' std', 0)
            for source, target in (('pooled_rho', 'rho'), ('pooled_pearson', 'pearson'), ('pooled_top1', 'top1_gt')):
                bounds = (None, None) if target == 'top1_gt' else (-1, 1)
                metrics[f'{cell}/pooled/{target}'] = _number(row[source], source, *bounds)
        return metrics
    if format_name == 'ckax_budget':
        native=_mapping(payload,format_name); protocol=_mapping(native['protocol'],'protocol')
        rows=_mapping(native['backbones'],'backbones'); _check_names(list(rows),expected_models)
        budgets=protocol['k_sweep']; total=protocol['n_candidates_N']
        splits=protocol['n_splits']
        if any(isinstance(value,bool) or not isinstance(value,int) or value<=0 for value in (total,splits)):
            raise ValueError('invalid CKA-X pool/split count')
        if not isinstance(budgets,list) or not budgets or any(
            isinstance(k,bool) or not isinstance(k,int) or not 0<k<total-1 for k in budgets) or len(set(budgets))!=len(budgets):
            raise ValueError('invalid CKA-X budget panel')
        for backbone,table in rows.items():
            _identifier(backbone); _mapping(table,'budget rows')
            if list(table)!=[str(k) for k in budgets]: raise ValueError('missing/reordered native budget cells')
            for k in budgets:
                row=_mapping(table[str(k)],'budget cell'); prefix=f'{backbone}/{k}labels'
                if row['backbone']!=backbone or row['k']!=k: raise ValueError('budget cell identity mismatch')
                valid=row['n_splits_valid']
                if isinstance(valid,bool) or not isinstance(valid,int) or not 0<valid<=protocol['n_splits']:
                    raise ValueError('invalid valid split count')
                for source,target in [('spearman','rho'),('pearson','pearson'),('top1','top1_gt')]:
                    summary=_mapping(row['fold_mean'][source],'fold_mean')
                    if isinstance(summary['n'],bool) or not isinstance(summary['n'],int) or not 0<summary['n']<=valid:
                        raise ValueError('invalid native per-metric split count')
                    bounds=(None,None) if source=='top1' else (-1,1)
                    metrics[f'{prefix}/fold_mean/{target}']=_number(summary['mean'],source,*bounds)
                    metrics[f'{prefix}/fold_std/{target}']=_number(summary['std'],source+' std',0)
                pooled=_mapping(row['pooled'],'pooled')
                if pooled['n_points']!=valid*(total-k): raise ValueError('native pooled point count disagrees with splits')
                for source,target in [('spearman','rho'),('pearson','pearson')]:
                    metrics[f'{prefix}/pooled/{target}']=_number(pooled[source],source,-1,1)
        return metrics
    if format_name in ('law_a', 'law_c'):
        rows = _mapping(payload, format_name)
        _check_names(list(rows), expected_models)
        for model, row in rows.items():
            _identifier(model); _mapping(row, model)
            if row.get('error'):
                raise ValueError(f'failed native model: {model}')
            if format_name == 'law_a':
                loss = _number(row['avg_loss'], 'avg_loss', 0)
                score = _number(row['a_score'], 'a_score')
                if score != -loss:
                    raise ValueError('A-score must be negative native mean NLL')
                metrics[f'{model}/mean_nll'] = loss
                metrics[f'{model}/a_score'] = score
            else:
                score = _number(row['c_score'], 'c_score', 0, 1)
                percent = _number(row['c_score_pct'], 'c_score_pct', 0, 100)
                if not math.isclose(percent, score * 100, rel_tol=0, abs_tol=1e-12):
                    raise ValueError('C-score fraction/percent disagree')
                metrics[f'{model}/pck10_fraction'] = score
                metrics[f'{model}/pck10_percent'] = percent
        return metrics
    if not isinstance(payload, list) or not payload:
        raise ValueError(f'{format_name} requires a non-empty native row list')
    names = []
    for row in payload:
        _mapping(row, 'row')
        model = _identifier(row['model' if format_name == 'metaclip2_zero_shot' else 'vision_encoder'])
        names.append(model)
        if format_name == 'metaclip2_zero_shot':
            percent = _number(row['top1_percent'], 'top1_percent', 0, 100)
            total, correct = row['images'], row['top1_correct']
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (total, correct)) or not 0 <= correct <= total or total <= 0:
                raise ValueError('invalid native prediction counts')
            if not math.isclose(percent, 100 * correct / total, rel_tol=0, abs_tol=1e-10):
                raise ValueError('native accuracy/counts disagree')
            metrics[f'{model}/top1_percent'] = percent
        else:
            if row['status'] == 'not applicable: no released reconstruction decoder':
                if any(row[key] != '' for key in ('t_acc_percent', 't_ned_percent', 'f_sim')):
                    raise ValueError('N/A decoder must not contain fabricated scores')
                continue
            if row['status'] != 'evaluated':
                raise ValueError(f'unaccepted TokBench status: {row["status"]}')
            for key, bounds in (('t_acc_percent', (0, 100)), ('t_ned_percent', (0, 100)), ('f_sim', (-1, 1))):
                try:
                    value = float(row[key]) if isinstance(row[key], str) else row[key]
                except ValueError as exc:
                    raise ValueError(f'invalid native {key}') from exc
                metrics[f'{model}/{key}'] = _number(value, key, *bounds)
    _check_names(names, expected_models)
    if not metrics:
        raise ValueError('native output contains no evaluated metrics')
    return metrics

def read_native_metrics(format_name,payload,*,expected_shots=None,expected_models=None):
    if format_name not in FORMATS:
        raise ValueError(f'unknown native metric format: {format_name}')
    if format_name != 'knn_multishot':
        return _other_metrics(format_name, payload, expected_models)
    if not isinstance(payload,list) or not payload:
        raise ValueError('knn_multishot requires a non-empty native row list')
    metrics,models,shots = {},set(),[]
    for row in payload:
        if not isinstance(row,dict) or not isinstance(row.get('Model'),str) or not row['Model'].strip():
            raise ValueError('native rows require an explicit nonempty model name')
        shot = row['Shot']
        if not isinstance(shot,int) or isinstance(shot,bool) or shot<=0 or shot in shots:
            raise ValueError('native shots must be unique positive integers')
        match = re.fullmatch(r'(\d+(?:\.\d+)?)%',str(row['Top1']))
        if not match or not 0<=float(match[1])<=100:
            raise ValueError('native Top1 requires a percentage in [0,100]')
        flops = row['TFLOPs']
        if isinstance(flops,bool) or not isinstance(flops,(int,float)) or not math.isfinite(flops) or flops<0:
            raise ValueError('native TFLOPs must be finite and nonnegative')
        models.add(row['Model']); shots.append(shot)
        metrics[f'top1_percent/{shot}shot'] = float(match[1])
        metrics[f'tflops/{shot}shot'] = float(flops)
    if len(models)!=1:
        raise ValueError('one unified run must contain exactly one native model')
    _check_names(list(models), expected_models)
    if expected_shots is not None and shots!=list(expected_shots):
        raise ValueError('native shot rows differ from the declared protocol')
    return metrics
