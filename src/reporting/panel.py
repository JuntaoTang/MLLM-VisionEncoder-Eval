"""Strict canonical 70-encoder tables, with explicit units and cell identity."""
import csv
import math
import re
from pathlib import Path

from ..core.artifacts import validate_result_payload, write_json_atomic
from ..core.hashing import sha256_json
from ..encoders import encoder_panel


def validate_cell(cell):
    if not isinstance(cell,dict) or set(cell)!={'encoder_id','column_id','metric'}:
        raise ValueError('report_cell needs explicit encoder_id, column_id and metric')
    if cell['encoder_id'] not in {spec.encoder_id for spec in encoder_panel()}:
        raise ValueError('report_cell encoder_id is not in canonical70')
    if not isinstance(cell['column_id'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cell['column_id']):
        raise ValueError('report_cell column_id must be a safe identifier')
    if not isinstance(cell['metric'],str) or not cell['metric'].strip():
        raise ValueError('report_cell metric must be explicit')


def validate_panel_spec(spec):
    if not isinstance(spec,dict) or spec.get('format')!='panel_table' or set(spec)-{'format','columns','allowed_na'}:
        raise ValueError('panel reporting requires explicit columns and optional declared N/A cells')
    columns=spec.get('columns')
    if not isinstance(columns,list) or not columns:
        raise ValueError('panel columns must be a nonempty ordered list')
    seen=set()
    for column in columns:
        if not isinstance(column,dict) or set(column)!={'id','method','dataset','unit'}:
            raise ValueError('each panel column needs id, method, dataset and unit')
        validate_cell({'encoder_id':encoder_panel()[0].encoder_id,'column_id':column['id'],'metric':'explicit'})
        if column['id'] in seen or any(not isinstance(column[key],str) or not column[key].strip() for key in ('method','dataset','unit')):
            raise ValueError('duplicate panel column or blank method/dataset/unit')
        seen.add(column['id'])
    na={}
    for record in spec.get('allowed_na',[]):
        if not isinstance(record,dict) or set(record)!={'encoder_id','column_id','reason'}:
            raise ValueError('N/A cells require identity and an explicit scientific reason')
        validate_cell({**{key:record[key] for key in ('encoder_id','column_id')},'metric':'not_applicable'})
        key=(record['encoder_id'],record['column_id'])
        if key in na or key[1] not in seen or not isinstance(record['reason'],str) or not record['reason'].strip():
            raise ValueError('duplicate/unknown N/A cell or missing reason')
        na[key]=record['reason']
    return columns,na


def write_panel_report(results,directory,spec):
    columns,na=validate_panel_spec(spec)
    panel=encoder_panel(); by_column={column['id']:column for column in columns}
    cells={}; provenance={}; run_ids=set()
    for result in results:
        validate_result_payload(result)
        if result['status']!='success': raise ValueError('unsuccessful panel result')
        if result['run_id'] in run_ids: raise ValueError('duplicate panel result run_id')
        run_ids.add(result['run_id'])
        cell=result['audit'].get('report_cell'); validate_cell(cell)
        key=(cell['encoder_id'],cell['column_id'])
        if key[1] not in by_column: raise ValueError('unexpected panel column')
        expected=by_column[key[1]]
        if result['method']!=expected['method'] or result['dataset']!=expected['dataset']:
            raise ValueError('panel cell method/dataset does not match its column')
        if key in cells or key in na: raise ValueError('duplicate panel cell or result conflicts with declared N/A')
        value=result['metrics'].get(cell['metric'])
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
            raise ValueError('panel cell metric is missing or nonfinite')
        # Known native unit tags are explicit schema facts, never guessed
        # from magnitude. Unknown custom metric units remain declarations.
        metric=cell['metric']
        unit=None
        if re.search(r'_percent(?:/|$)',metric): unit='percent'
        elif re.search(r'_fraction(?:/|$)',metric): unit='fraction'
        elif '/legacy_display_score/' in metric: unit='legacy_display_score'
        elif metric.startswith('tflops/'): unit='tflops'
        if unit is not None and expected['unit']!=unit:
            raise ValueError('panel native metric unit differs from declared column unit')
        if unit=='percent' and not 0<=value<=100 or unit=='fraction' and not 0<=value<=1:
            raise ValueError('panel native metric outside its declared range')
        cells[key]=value
        provenance[key]={'run_id':result['run_id'],'metric':cell['metric'],'protocol':result['protocol'],'audit':result['audit']}
    expected={(model.encoder_id,column['id']) for model in panel for column in columns}
    if set(cells)|set(na)!=expected:
        missing=sorted(expected-set(cells)-set(na))
        raise ValueError(f'panel incomplete; missing {len(missing)} explicit cells: {missing[:5]}')
    rows=[]; records=[]
    for model in panel:
        row={'rank':model.rank,'encoder_id':model.encoder_id}
        for column in columns:
            key=(model.encoder_id,column['id']); row[column['id']]=cells.get(key,'N/A')
            records.append({'encoder_id':key[0],'column_id':key[1],'unit':column['unit'],
                            'value':cells.get(key),'status':'not_applicable' if key in na else 'success',
                            'reason':na.get(key),'provenance':provenance.get(key)})
        rows.append(row)
    # Validate every cell before publishing anything; missing is never zero.
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'panel_spec_sha256':sha256_json(spec),'columns':columns,'rows':rows,'cells':records,
             'scope':'Verified canonical70 table for explicitly declared inputs/columns, not full-data accuracy or all acceptance gates'}
    write_json_atomic(directory/'panel.json',payload)
    with (directory/'panel.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['rank','encoder_id']+[column['id'] for column in columns]); writer.writeheader(); writer.writerows(rows)
    def escape(value): return str(value).replace('|','\\|').replace('\n',' ')
    labels=['Rank','Encoder']+[escape(f"{column['id']} ({column['unit']})") for column in columns]
    lines=['# Reproducibility report','',payload['scope']+'.','',
           f'Complete: {len(cells)} numeric cells; {len(na)} explicitly declared N/A cells. No averaging or score imputation.','',
           '| '+' | '.join(labels)+' |','|'+'---|'*len(labels)]
    for row in rows: lines.append('| '+' | '.join(str(value) for value in row.values())+' |')
    lines.extend(['','Per-cell config/code/input/weight/environment hashes and protocol are retained in `panel.json`.','',
                  'G5 full-data reproduction is not established by this table.',''])
    (directory/'REPRODUCIBILITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return payload
