"""JSON boundary for isolated model workers; never invokes a shell."""
import argparse
import runpy
import sys

from ..core.artifacts import read_json, write_json_atomic
from .registry import get_worker

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request',required=True)
    p.add_argument('--result',required=True)
    args = p.parse_args()
    request = read_json(args.request)
    if request.get('schema_version') != 1:
        p.error('request.schema_version must be 1')
    spec = get_worker(request['worker'])
    if spec.blocked:
        p.error(spec.blocked)
    argv = request.get('arguments',[])
    if not isinstance(argv,list) or not all(isinstance(item,str) for item in argv):
        p.error('request.arguments must be an argv string list')
    result = {'schema_version':1,'worker':request['worker'],'status':'failed','returncode':1}
    try:
        sys.argv = [spec.module,*argv]
        runpy.run_module(spec.module,run_name='__main__',alter_sys=True)
        result.update(status='success',returncode=0)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code,int) else (0 if exc.code is None else 1)
        if code and not isinstance(exc.code,(int,type(None))):
            print(str(exc.code),file=sys.stderr)
        result.update(status='success' if code==0 else 'failed',returncode=code)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    finally:
        write_json_atomic(args.result,result)
    return result['returncode']

if __name__ == '__main__':
    raise SystemExit(main())
