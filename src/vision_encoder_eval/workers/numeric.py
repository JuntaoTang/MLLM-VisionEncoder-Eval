"""Resolved JSON protocol for a numerical method in its configured interpreter."""
import argparse

from ..core.artifacts import read_json,write_json_atomic,validate_result_payload
from ..core.registry import METHODS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request',required=True)
    p.add_argument('--result',required=True)
    args = p.parse_args()
    resolved = read_json(args.request)
    spec = METHODS.get(resolved['experiment']['method'])
    if spec.execution_mode!='in_process':
        p.error('numerical worker accepts only in-process numerical methods')
    result = spec.runner(resolved)
    validate_result_payload(result)
    write_json_atomic(args.result,result)

if __name__=='__main__': main()
