#!/usr/bin/env python3
"""Snapshot and submit reconstruction/post-training, or an unprimed baseline."""
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
    parser.add_argument('--queue', default='dbeigpu')
    parser.add_argument('--hours', type=int, default=48)
    parser.add_argument('--submit', action='store_true', help='Actually submit; default stages files only')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Submit only a fresh 250000-update unprimed MMD-flow baseline')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    study = new_run_path(args.study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    source = study / 'source'
    source.mkdir()
    for path in list(repo.glob('*.py')) + list(repo.glob('*LICENSE*')):
        shutil.copy2(path, source / path.name)
    shutil.copytree(repo / 'scripts/lsf', source / 'scripts/lsf', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo / 'tests', source / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo / 'configs', source / 'configs')
    (source / 'assets').symlink_to(repo / 'assets', target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    (source / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    worker = source / 'scripts/lsf/run_cifar_priming.sh'
    manifest = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    paper='https://arxiv.org/abs/2610.20331', study=str(study),
                    recipe=('250000 distributional updates from random initialization'
                            if args.baseline_only else
                            '80 reconstruction epochs followed by 250000 distributional updates'),
                    queue=args.queue, jobs={})
    manifest_path = study / 'deployment.json'
    for stage in (('baseline',) if args.baseline_only else ('reconstruction', 'posttraining')):
        output = study / stage
        output.mkdir()
        command = ['bsub', '-q', args.queue, '-J', f'cifar_priming_{stage}', '-n', '4',
                   '-R', 'span[hosts=1] rusage[mem=32GB]',
                   '-gpu', 'num=1:mode=exclusive_process:gmem=80G',
                   '-W', f'{args.hours}:00', '-oo', str(output / 'scheduler.log')]
        if stage == 'posttraining':
            pre_id = manifest['jobs']['reconstruction'].get('job_id', 'RECONSTRUCTION_JOB_ID')
            command += ['-w', f'done({pre_id})']
        command += ['env', f'PYTHON_BIN={repo / ".venv/bin/python"}', f'DATA_ROOT={repo / "data"}',
                    'bash', str(worker), stage, str(study)]
        record = dict(command=command, output=str(output))
        if args.submit:
            result = subprocess.run(command, text=True, capture_output=True)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            match = re.search(r'Job <(\d+)> is submitted', result.stdout)
            if result.returncode == 0 and match:
                record['job_id'] = match.group(1)
            else:
                manifest['jobs'][stage] = record
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                raise RuntimeError(f'Submission failed; earlier jobs are recorded in {manifest_path}: {result.stderr}')
        manifest['jobs'][stage] = record
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({stage: record}), flush=True)


if __name__ == '__main__':
    main()
