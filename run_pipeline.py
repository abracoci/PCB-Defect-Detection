"""Run either preserved preprocessing engine with a fresh output folder."""
import argparse, csv, subprocess, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['original','flexible'], required=True)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--margin', type=float, default=.10)
    a = p.parse_args(); src=a.input.resolve(); dest=a.output.resolve()
    if not src.exists(): p.error('Input does not exist')
    if dest.exists(): p.error('Output already exists; select a new folder')
    if src == dest or src in dest.parents or dest in src.parents: p.error('Input and output must be separate')
    if dest == BASE or dest in BASE.parents: p.error('Output cannot replace package')
    def run(folder, script, args):
        return subprocess.run([sys.executable,'-u',str(BASE/folder/script),*map(str,args)],cwd=BASE/folder).returncode
    if a.mode == 'original':
        if not (src/'metadata.csv').is_file() or not (src/'raw').is_dir(): p.error('Original requires dataset root containing raw/ and metadata.csv')
        args=['--root',src,'--out',dest]
        for script, extra in [('stage1_dataset.py',['--check-images']),('stage3a_probe.py',[]),('stage3b_export.py',[])]:
            rc=run('original',script,args+extra)

            if script == 'stage3a_probe.py' and rc == 3:
                calibration = dest / '_stage3' / 'calibration_good_only.json'

                if calibration.is_file():
                    print("QC warnings detected. Continuing to export.")
                    continue

            if rc:
                return rc
        return 0
    rc=run('flexible','pcb_flexible_preprocess_v2.py',['--input',src,'--output',dest,'--size',a.size,'--margin',a.margin]+(['--debug'] if a.debug else []))
    if rc: return rc
    with (dest/'report.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    return 0 if rows and all(r['status']=='OK' for r in rows) else 1
if __name__=='__main__': raise SystemExit(main())
