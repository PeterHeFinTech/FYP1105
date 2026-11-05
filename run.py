"""Single entry point for HMDA preparation, models, evaluation and checks."""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    args = sys.argv[1:]
    usage = '''Usage: python run.py COMMAND [OPTIONS]
  benchmark FILE [FILE ...]    Compare CSVs with the fixed full HMDA reference
  prepare                      Prepare a new split (existing data is protected)
  tabddpm train|sample          Train or sample TabDDPM
  ctgan train|sample            Train or sample CTGAN
  evaluate tabddpm|ctgan        Evaluate that model's generated CSV
  check tabddpm|ctgan           Run functional checks
Append --help to a command for its options.
Default paths are relative to the project, even from another working directory.
Explicit paths are relative to your current working directory.
'''
    if not args or args[0] in ['-h', '--help']:
        print(usage)
        return
    command, *rest = args
    if command == 'benchmark':
        module, forwarded = 'evaluation.benchmark', rest
    elif command == 'prepare':
        module, forwarded = 'hmda', ['prepare', *rest]
    elif command in ['tabddpm', 'ctgan'] and rest and rest[0] in ['train', 'sample']:
        module = 'hmda' if command == 'tabddpm' else 'hmda_ctgan'
        forwarded = rest
    elif command == 'evaluate' and rest and rest[0] in ['tabddpm', 'ctgan']:
        model, *options = rest
        module = 'hmda'
        forwarded = ['evaluate', '--synthetic', str(ROOT/'outputs'/model/'synthetic.csv'),
                     '--out', str(ROOT/'outputs'/model/'evaluation'), *options]
    elif command == 'check' and rest and rest[0] in ['tabddpm', 'ctgan']:
        sys.path.insert(0, str(ROOT/'tests'))
        module = 'check' if rest[0] == 'tabddpm' else 'check_ctgan'
        forwarded = rest[1:]
    else:
        raise SystemExit(usage)
    sys.argv = [f'run.py {command}', *forwarded]
    importlib.import_module(module).main()


if __name__ == '__main__':
    main()
