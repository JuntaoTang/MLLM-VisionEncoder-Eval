"""Explicit paired-cache manifest -> complete canonical70 numerical suites."""
from pathlib import Path
import re

from .loader import _load_document
from .schema import ConfigError
from ..core.artifacts import write_json_atomic
from ..encoders import encoder_panel


def generate_feature_panel(manifest_path,output):
    import numpy as np
    from ..data.features import validate_paired_rows
    manifest_path=Path(manifest_path).resolve(); output=Path(output).resolve()
    if output.exists(): raise ConfigError('output already exists; reviewed configs must not be overwritten')
    document=_load_document(manifest_path)
    if set(document)!={'schema_version','dataset','visual','text','methods'} or document['schema_version']!=1:
        raise ConfigError('feature panel manifest needs schema_version, dataset, visual, text and methods')
    if not isinstance(document['dataset'],str) or not document['dataset'].strip(): raise ConfigError('explicit panel dataset required')
    panel=encoder_panel(); visual=document['visual']; texts=document['text']; methods=document['methods']
    if not isinstance(visual,dict) or set(visual)!={model.encoder_id for model in panel}:
        raise ConfigError('visual manifest must contain each canonical70 encoder exactly once')
    if not isinstance(texts,dict) or not texts or not isinstance(methods,dict) or not methods:
        raise ConfigError('text encoders and methods must be explicitly declared')
    if not set(methods).issubset({'ravel','rsa','cca','gw','mutualnn'}): raise ConfigError('feature panel supports only the five paired numerical methods')
    for name in texts:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name): raise ConfigError('unsafe text encoder ID')
    for protocol in methods.values():
        if not isinstance(protocol,dict) or protocol.get('assume_aligned_rows'):
            raise ConfigError('panel methods require protocol mappings and validated sample IDs')
    def files(record):
        if not isinstance(record,dict) or set(record)!={'features','manifest'}:
            raise ConfigError('each feature entry needs explicit features and row manifest paths')
        result={}
        for key,value in record.items():
            if not isinstance(value,str) or not value.strip(): raise ConfigError('feature paths must be explicit')
            path=Path(value).expanduser()
            path=(manifest_path.parent/path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file(): raise ConfigError(f'feature panel input does not exist: {path}')
            result[key]=str(path)
        return result
    required_representations={'patch' if method=='ravel' else 'global' for method in methods}
    visual_files={}
    for name,record in visual.items():
        if isinstance(record,dict) and set(record)=={'features','manifest'}:
            if len(required_representations)!=1:
                raise ConfigError('mixed RAVEL/global methods require explicit global and patch caches')
            representation=next(iter(required_representations))
            visual_files[name]={representation:files(record)}
        else:
            if not isinstance(record,dict) or not set(record).issubset({'global','patch'}) or not required_representations.issubset(record):
                raise ConfigError('visual entries require the selected global and/or patch representations')
            visual_files[name]={representation:files(entry) for representation,entry in record.items()}
    text_files={name:files(record) for name,record in texts.items()}
    # Check all sample identities, content checksums and numerical input shapes
    # before writing a single config. Dimension differences remain permitted.
    for model in panel:
        for representation,vf in visual_files[model.encoder_id].items():
            va=np.load(vf['features'],mmap_mode='r',allow_pickle=False)
            rank=3 if representation=='patch' else 2
            if va.ndim!=rank or va.dtype.kind not in 'fiu' or not np.isfinite(va).all():
                raise ConfigError(f'{representation} visual arrays require finite numeric rank-{rank} features')
            for text_name,tf in text_files.items():
                ta=np.load(tf['features'],mmap_mode='r',allow_pickle=False)
                if ta.ndim!=2 or ta.dtype.kind not in 'fiu' or not np.isfinite(ta).all():
                    raise ConfigError('text panel arrays require finite numeric [N,D] features')
                validate_paired_rows({'visual_manifest':vf['manifest'],'text_manifest':tf['manifest']},
                                    vf['features'],tf['features'],va,ta,{})
    output.mkdir(parents=True)
    names=[]; columns=[]
    for method,protocol in methods.items():
        for text_name,tf in text_files.items():
            column=f'{method}__{text_name}'
            columns.append({'id':column,'method':method,'dataset':document['dataset'],'unit':'unitless'})
            for model in panel:
                vf=visual_files[model.encoder_id]['patch' if method=='ravel' else 'global']
                name=f'{model.rank:03d}_{model.encoder_id}__{column}'
                filename=name+'.json'; names.append(filename)
                value={'schema_version':1,'experiment':{'name':name,'kind':'method'},'method':method,
                    'dataset':document['dataset'],'protocol':protocol,
                    'report_cell':{'encoder_id':model.encoder_id,'column_id':column,'metric':'final_score'},
                    'inputs':{'visual_features':vf['features'],'visual_manifest':vf['manifest'],
                              'text_features':tf['features'],'text_manifest':tf['manifest'],
                              'panel_manifest':str(manifest_path)}}
                write_json_atomic(output/filename,value)
    write_json_atomic(output/'suite.json',{'schema_version':1,'experiment':{'name':'paired_feature_panel70','kind':'suite'},
        'experiments':names,'reporting':{'format':'panel_table','columns':columns}})
    return {'suite':str(output/'suite.json'),'models':70,'columns':len(columns),'runs':len(names),
            'scope':'Explicit validated paired cached features; not encoder extraction or full-paper accuracy'}
