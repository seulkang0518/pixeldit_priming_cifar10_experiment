#!/usr/bin/env python3
"""Snapshot the DDPM sanity check and optionally submit one LSF GPU job."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_storage import new_run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    repo = Path(__file__).resolve().parents[2]
    study = new_run_path(args.study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    source = study / 'source'
    source.mkdir()
    for name in ('ddpm.py', 'backbone.py', 'data.py', 'fid.py', '_fid_inception.py', 'visualization.py', 'run_storage.py'):
        shutil.copy2(repo / name, source / name)
    shutil.copytree(repo / 'scripts/lsf', source / 'scripts/lsf', ignore=shutil.ignore_patterns('__pycache__'))
    (source / 'tests').mkdir()
    shutil.copy2(repo / 'tests/test_ddpm.py', source / 'tests/test_ddpm.py')
    (source / 'assets').symlink_to(repo / 'assets', target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    (source / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    output = study / 'training'
    output.mkdir()
    command = ['bsub', '-q', 'dbeigpu', '-J', 'cifar_ddpm_sanity', '-n', '4',
               '-R', 'span[hosts=1] rusage[mem=32GB]', '-gpu', 'num=1:mode=exclusive_process:gmem=80G',
               '-W', '120:00', '-oo', str(output / 'scheduler.log'),
               'env', f'PYTHON_BIN={repo / ".venv/bin/python"}', f'DATA_ROOT={repo / "data"}',
               'bash', str(source / 'scripts/lsf/run_cifar_ddpm.sh'), str(study), str(args.steps)]
    record = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='DDPM architecture sanity check', steps=args.steps, command=command,
                  source=str(source), output=str(output))
    if args.submit:
        result = subprocess.run(command, text=True, capture_output=True)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        match = re.search(r'Job <(\d+)> is submitted', result.stdout)
        if result.returncode == 0 and match:
            record['job_id'] = match.group(1)
        else:
            (study / 'deployment.json').write_text(json.dumps(record, indent=2) + '\n')
            raise RuntimeError(f'Submission failed: {result.stderr}')
    (study / 'deployment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
