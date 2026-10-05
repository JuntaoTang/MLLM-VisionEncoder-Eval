"""Generate a 70-model cached-feature CPU suite, not raw encoder exports."""
from pathlib import Path
import numpy as np

from .loader import _load_document,_expand_environment,_resolve_local_paths
from .schema import validate_local_config,ConfigError
from ..core.artifacts import read_json,write_json_atomic
from ..encoders import encoder_panel

def generate_knn_panel(local_path,output):
    from ..workers.knn.different_shot.models import ijepa
    from ..workers.knn.different_shot.run_preexported_features import SPECS
    local_path = Path(local_path).resolve()
    local = dict(_expand_environment(_load_document(local_path)))
    validate_local_config(local)
    _resolve_local_paths(local,config_dir=local_path.parent)
    root = Path(local['paths'].get('knn_exports',''))
    if not local['paths'].get('knn_exports') or not root.is_dir():
        raise ConfigError('configure local.paths.knn_exports to the shared exported feature directory')
    output = Path(output).resolve()
    if output.exists():
        raise ConfigError('output directory already exists; do not overwrite reviewed configs')
    entries = read_json(root/'export_manifest.json').get('entries',[])
    panel = encoder_panel()
    by_token = {entry['tokenizer']:entry for entry in entries}
    if len(entries)!=70 or len(by_token)!=70 or set(by_token)!={p.encoder_id for p in panel}:
        raise ConfigError('export manifest must match the unique canonical 70-encoder panel')
    labels = np.load(root/'labels.npy',allow_pickle=False)
    source = np.load(root/'source_indices.npy',allow_pickle=False)
    if (labels.shape!=(200000,) or source.shape!=labels.shape or labels.dtype.kind not in 'iu'
            or source.dtype.kind not in 'iu' or source[0]<0 or not np.all(source[1:]>source[:-1])):
        raise ConfigError('requires the shared 200000-row export with increasing original source indices')
    for spec in panel:
        entry = by_token[spec.encoder_id]
        relative = Path(entry['feature_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ConfigError('export features must be safe relative paths')
        array = np.load(root/relative,mmap_mode='r',allow_pickle=False)
        if entry['rank']!=spec.rank or array.shape!=(200000,entry['feature_dim']) or array.dtype!=np.float32:
            raise ConfigError(f'export shape/dtype/rank mismatch: {spec.encoder_id}')
    protocol = ijepa.build_protocol(labels,source)
    protocol.update(dataset=str(root),labels_file=str(root/'labels.npy'),
                    source_indices_file=str(root/'source_indices.npy'))
    protocol.pop('feature_file',None)
    output.mkdir(parents=True)
    write_json_atomic(output/'protocol.json',protocol)
    experiments = []
    for spec in panel:
        entry = by_token[spec.encoder_id]
        display,flops = SPECS[spec.encoder_id]
        filename = f'{spec.rank:03d}_{spec.encoder_id}.json'
        experiments.append(filename)
        feature = '${paths.knn_exports}/'+entry['feature_file']
        config = {'schema_version':1,'experiment':{'name':'knn_export_'+spec.encoder_id,'kind':'method'},
            'method':'knn','dataset':'imagenet_train_export200_seed42_195pool_5query',
            'inputs':{'features':feature,'labels':'${paths.knn_exports}/labels.npy',
                'source_indices':'${paths.knn_exports}/source_indices.npy',
                'export_manifest':'${paths.knn_exports}/export_manifest.json','protocol':'protocol.json'},
            'protocol':{'train_shots':list(ijepa.TRAIN_SHOTS),'k':20,'temperature':0.07,
                'num_classes':1000,'normalization':'L2','backend':'FAISS CPU IndexFlatIP',
                'encoder_id':spec.encoder_id,'rank':spec.rank,
                'feature_representation':entry['representation']},
            'steps':[{'worker':'knn.different_shot.evaluate_features',
                'arguments':['--features',feature,'--labels','${paths.knn_exports}/labels.npy',
                    '--protocol',str(output/'protocol.json'),'--model',display,
                    '--flops-per-image',flops,'--output','${workspace}/scores'],
                'outputs':['${workspace}/scores.json','${workspace}/scores.csv'],
                'metrics_adapter':{'format':'knn_multishot','path':'${workspace}/scores.json'}}]}
        write_json_atomic(output/filename,config)
    write_json_atomic(output/'suite.json',{'schema_version':1,
        'experiment':{'name':'knn_cached_panel70','kind':'suite'},'experiments':experiments,
        'reporting':{'format':'long_table'}})
    return {'suite':str(output/'suite.json'),'models':len(experiments),
            'shots':list(ijepa.TRAIN_SHOTS),'backend':'FAISS CPU IndexFlatIP',
            'scope':'CPU native cached-feature protocol; not raw encoder export or GPU backend parity'}
