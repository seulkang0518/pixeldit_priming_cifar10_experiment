"""Snapshot a compact DiT comparison; optionally submit its two-stage GPU job."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_storage import new_run_path

from run_cifar_pair import set_option, write_json


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=repo / 'results/cifar/cadence_2500_ckpt20000_20261026/corrected_riesz.json')
    parser.add_argument('--queue', default='dbeigpu')
    parser.add_argument('--hours', type=int, default=48)
    parser.add_argument('--lpips-weight', type=float, default=0.0,
                        help='Add frozen VGG LPIPS to reconstruction only')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.lpips_weight) or args.lpips_weight < 0:
        parser.error('--lpips-weight must be finite and nonnegative')
    reference = json.loads(args.reference.read_text())
    post = list(reference['run']['training_command'])
    assert post[post.index('--objective') + 1] == 'softmax-riesz'
    assert '--no-field-rms-normalization' in post and '--no-normalize-per-temperature' in post
    study = new_run_path(args.study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    source = study / 'source'
    source.mkdir()
    for path in [*repo.glob('*.py'), *repo.glob('*LICENSE*')]:
        shutil.copy2(path, source / path.name)
    for name in ('scripts/lsf', 'tests'):
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    (source / 'assets').symlink_to(repo / 'assets', target_is_directory=True)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    write_json(source / 'sha256.json', hashes)
    python = str(repo / '.venv/bin/python')
    reconstruction = study / 'reconstruction'
    posttraining = study / 'posttraining'
    reconstruction.mkdir()
    posttraining.mkdir()
    dit = ['--architecture', 'dit', '--dit-width', '384', '--dit-depth', '12',
           '--dit-heads', '6', '--dit-patch-size', '2']
    pre = [python, '-u', str(source / 'main.py'), '--dataset', 'cifar10',
           '--data-root', str(repo / 'data'), '--output', str(reconstruction),
           '--target-size', '0', '--particles', '64', '--eval-size', '512',
           '--target-batch-size', '64', '--training-stage', 'reconstruction',
           '--corruption-level', '0.8', '--epochs', '80', '--learning-rate', '0.0002',
           '--optimizer', 'Adam', '--adam-beta2', '0.999', '--ema-decay', '0.9999',
           '--save-every', '20', '--checkpoint-every', '1', '--fid-samples', '10000',
           '--fid-batch-size', '64', '--trajectory-particles', '8', '--seed', '42',
           '--threads', '4', '--device', 'cuda', *dit]
    smoke = [python, '-u', str(source / 'scripts/lsf/smoke_dit.py'),
             str(repo / 'data'), str(study / 'smoke.json')]
    if args.lpips_weight:
        perceptual_options = ['--lpips-weight', str(args.lpips_weight),
                              '--lpips-cache', str(study / 'perceptual_cache')]
        pre += perceptual_options
        smoke += perceptual_options
    post[:3] = [python, '-u', str(source / 'main.py')]
    for flag in ('--resume', '--fork-checkpoint'):
        if flag in post:
            index = post.index(flag)
            del post[index:index + 2]
    for flag, value in [('--output', posttraining), ('--init-checkpoint', reconstruction / 'checkpoint_latest.pt'),
                        ('--data-root', repo / 'data'), ('--threads', 4), *zip(dit[::2], dit[1::2])]:
        set_option(post, flag, value)
    manifest = dict(study=str(study), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    reference_manifest=str(args.reference.resolve()),
                    comparison=('DiT MSE plus frozen VGG LPIPS priming; pixel-only corrected Riesz post-training'
                                if args.lpips_weight else
                                'DiT architecture with independent reconstruction priming; corrected no-RMS softmax Riesz recipe retained'),
                    pretraining_lpips_weight=args.lpips_weight,
                    smoke_command=smoke,
                    stages=[dict(stage='reconstruction', output=str(reconstruction), command=pre),
                            dict(stage='posttraining', output=str(posttraining), command=post)])
    manifest_path = study / 'manifest.json'
    write_json(manifest_path, manifest)
    job_name = 'cifar_dit_lpips_riesz' if args.lpips_weight else 'cifar_compact_dit_riesz'
    command = ['bsub', '-q', args.queue, '-J', job_name, '-n', '4',
               '-R', 'span[hosts=1] rusage[mem=32GB]', '-gpu', 'num=1:mode=exclusive_process:gmem=80G',
               '-W', f'{args.hours}:00', '-oo', str(study / 'scheduler.log'),
               python, '-u', str(source / 'scripts/lsf/run_compact_dit.py'), '--manifest', str(manifest_path)]
    deployment = dict(status='staged', command=command, manifest=str(manifest_path))
    write_json(study / 'deployment.json', deployment)
    if args.submit:
        result = subprocess.run(command, text=True, capture_output=True)
        deployment.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        match = re.search(r'Job <(\d+)> is submitted', result.stdout)
        deployment.update(status='submitted' if result.returncode == 0 and match else 'submission_failed')
        if match:
            deployment['job_id'] = match.group(1)
        write_json(study / 'deployment.json', deployment)
        if deployment['status'] != 'submitted':
            raise RuntimeError(f'Submission failed: {result.stderr}')
    print(json.dumps(deployment, indent=2), flush=True)


if __name__ == '__main__':
    main()
