"""Row-identity and content contracts for paired feature arrays."""
import json
from pathlib import Path

from ..core.hashing import sha256_file
from ..core.artifacts import write_json_atomic

def create_feature_manifest(array_path, sample_ids, output):
    import numpy as np
    array = np.load(array_path,mmap_mode='r',allow_pickle=False)
    value = {'schema_version':1,'shape':list(array.shape),'dtype':str(array.dtype),
             'sha256':sha256_file(array_path),'sample_ids':sample_ids}
    if not isinstance(sample_ids,list) or not all(isinstance(v,str) and v for v in sample_ids) or len(sample_ids)!=len(array) or len(set(sample_ids))!=len(sample_ids):
        raise ValueError('sample IDs must be a unique list in exact feature-row order')
    write_json_atomic(output,value)
    return value

def validate_feature_manifest(manifest_path, array_path, array):
    value = json.loads(Path(manifest_path).read_text())
    if value.get('schema_version') != 1:
        raise ValueError('feature manifest schema_version must be 1')
    if value.get('sha256') != sha256_file(array_path):
        raise ValueError(f'feature manifest content hash mismatch: {array_path}')
    if value.get('shape') != list(array.shape) or value.get('dtype') != str(array.dtype):
        raise ValueError('feature manifest shape/dtype mismatch')
    ids = value.get('sample_ids')
    if not isinstance(ids,list) or not all(isinstance(v,str) and v for v in ids) or len(ids)!=len(array) or len(set(ids))!=len(ids):
        raise ValueError('feature manifest needs unique sample IDs for every row')
    return ids

def validate_paired_rows(inputs, visual_path, text_path, visual, text, protocol):
    names = ['visual_manifest','text_manifest']
    present = [name in inputs for name in names]
    if any(present) and not all(present):
        raise ValueError('both visual_manifest and text_manifest are required together')
    if all(present):
        left = validate_feature_manifest(inputs[names[0]],visual_path,visual)
        right = validate_feature_manifest(inputs[names[1]],text_path,text)
        if left != right:
            raise ValueError('visual/text sample IDs or row order differ')
        return 'verified_sample_ids_and_content'
    if protocol.get('assume_aligned_rows') is not True:
        raise ValueError('paired features require row-ID manifests, or explicit protocol.assume_aligned_rows=true for fixtures/legacy exports')
    return 'explicit_assumption_not_verified'
