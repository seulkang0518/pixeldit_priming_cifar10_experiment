#!/usr/bin/env python3
"""Freeze a two-stage CIFAR backbone experiment, then run/resume its manifest."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from run_storage import new_run_path


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def set_option(command, flag, value):
    if flag in command:
        index = command.index(flag)
        command[index + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def default_commands(python, source, study, data_root):
    """Documented compact-DiT recipe; NOT an asserted reproduction of FID 65."""
    common = [python, '-u', str(source / 'main.py'), '--dataset', 'cifar10',
              '--data-root', str(data_root), '--target-size', '0', '--particles', '64',
              '--eval-size', '512', '--fid-samples', '10000', '--fid-batch-size', '64',
              '--trajectory-particles', '8', '--seed', '42', '--threads', '4', '--device', 'cuda',
              '--optimizer', 'Adam', '--adam-beta2', '.999', '--ema-decay', '.9999']
    pre = common + ['--output', str(study / 'reconstruction'), '--training-stage', 'reconstruction',
                    '--corruption-level', '.8', '--epochs', '80', '--target-batch-size', '64',
                    '--learning-rate', '.0002', '--save-every', '20', '--checkpoint-every', '1']
    post = common + ['--output', str(study / 'posttraining'), '--training-stage', 'distributional',
                     '--steps', '250000', '--target-batch-size', '256', '--train-particles', '128',
                     '--objective', 'softmax-riesz', '--no-field-rms-normalization',
                     '--no-normalize-per-temperature', '--tau-list', '.2', '.05', '.02',
                     '--multiscale-loss', '--coarse-sizes', '8', '16', '--patch-sizes', '7', '14',
                     '--crops-per-size', '4', '--global-loss-weight', '2', '--coarse-loss-weight', '1',
                     '--patch-loss-weight', '1', '--random-flip', '--max-grad-norm', '1',
                     '--learning-rate', '.0002', '--min-learning-rate', '.0001',
                     '--lr-schedule', 'cosine', '--warmup-steps', '1000',
                     '--save-every', '2500', '--checkpoint-every', '1000',
                     '--keep-checkpoint-every', '20000']
    return pre, post


def prepare(args):
    study = new_run_path(args.study).resolve()
    if study.exists():
        raise FileExistsError(f'{study} already exists; use run --manifest to resume it')
    # Resolve the reference before creating output, so a bad reference leaves no partial study.
    reference = json.loads(args.reference.read_text()) if args.reference else None
    weights = REPO / 'assets/pt_inception-2015-12-05-6726825d.pth'
    expected = (REPO / 'assets/inception.sha256').read_text().split()[0]
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Run python scripts/cs_cluster/prepare_pixeldit_assets.py before freezing the study')
    source = study / 'source'
    pre, post = default_commands(args.python, source, study, args.data_root.resolve())
    recipe_status = 'documented_recipe; exact FID-65 run manifest unavailable'
    if reference is not None:
        if 'stages' in reference:
            commands = {row['stage']: list(row['command']) for row in reference['stages']}
            pre, post = commands['reconstruction'], commands['posttraining']
            recipe_status = 'both stage commands copied from supplied manifest'
        else:
            post = list(reference['run']['training_command'])
            recipe_status = 'post command copied from reference; reconstruction uses documented MSE recipe'
    for name, command in [('reconstruction', pre), ('posttraining', post)]:
        command[:3] = [args.python, '-u', str(source / 'main.py')]
        for flag in ('--resume', '--fork-checkpoint', '--init-checkpoint'):
            if flag in command:
                index = command.index(flag)
                del command[index:index + 2]
        for flag, value in [('--data-root', args.data_root.resolve()), ('--output', study / name),
                            ('--device', 'cuda'), ('--threads', 4), ('--architecture', args.backbone)]:
            set_option(command, flag, value)
        if '--lpips-cache' in command:
            set_option(command, '--lpips-cache', study / 'perceptual_cache')
        if args.data_parallel and '--data-parallel' not in command:
            command.append('--data-parallel')
        if not args.data_parallel and '--data-parallel' in command:
            command.remove('--data-parallel')
    set_option(post, '--init-checkpoint', study / 'reconstruction/checkpoint_latest.pt')
    study.mkdir(parents=True)
    source.mkdir()
    for file in [*REPO.glob('*.py'), *REPO.glob('*LICENSE*')]:
        shutil.copy2(file, source / file.name)
    for directory in ('third_party', 'scripts/cs_cluster', 'tests', 'assets'):
        shutil.copytree(REPO / directory, source / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {str(file.relative_to(source)): hashlib.sha256(file.read_bytes()).hexdigest()
              for file in source.rglob('*') if file.is_file()}
    write_json(source / 'sha256.json', hashes)
    stages = [dict(stage=name, output=str(study / name), command=command)
              for name, command in [('reconstruction', pre), ('posttraining', post)]]
    for stage in stages:
        Path(stage['output']).mkdir()
    smoke = [args.python, '-u', str(source / 'scripts/cs_cluster/smoke_pixeldit.py'),
             '--backbone', args.backbone, '--output', str(study / 'smoke.json')]
    if args.data_parallel:
        smoke.append('--data-parallel')
    if '--lpips-weight' in pre:
        smoke.extend(['--lpips-weight', pre[pre.index('--lpips-weight') + 1],
                      '--lpips-cache', str(study / 'perceptual_cache')])
    manifest = dict(study=str(study), backbone=args.backbone, recipe_status=recipe_status,
                    reference=str(args.reference.resolve()) if args.reference else None,
                    reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
                    upstream_baseline='hudsonchen/mmd_flow_hudson_new@12213c3d9889b441e520b6fe375fe001f48b9ce6',
                    smoke_command=smoke, stages=stages,
                    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(study / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if args.dry_run:
        for command in [manifest['smoke_command'], *(row['command'] for row in manifest['stages'])]:
            print(shlex.join(command))
        return
    study = Path(manifest['study'])
    child = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Refuse altered source/weights when resuming the frozen experiment.
    source = study / 'source'
    for name, digest in json.loads((source / 'sha256.json').read_text()).items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen source checksum mismatch: {name}')
    with (study / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (study / 'smoke.log').open('a', buffering=1) as stream:
            child = subprocess.Popen(manifest['smoke_command'], stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait():
                raise RuntimeError(f'GPU capacity smoke failed; inspect {study / "smoke.log"}')
        if args.smoke_only:
            print(f'Smoke passed: {study / "smoke.json"}')
            return
        for stage in manifest['stages']:
            output = Path(stage['output'])
            status_path = output / 'job_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status') == 'complete':
                continue
            command = list(stage['command'])
            checkpoint = output / 'checkpoint_latest.pt'
            if checkpoint.exists():
                set_option(command, '--resume', checkpoint)
            status = dict(status='running', stage=stage['stage'], command=command,
                          job_id=os.environ.get('JOB_ID'), host=os.uname().nodename)
            with (output / 'run.log').open('a', buffering=1) as stream:
                stream.write(shlex.join(command) + '\n')
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(status_path, status)
                try:
                    code = child.wait()
                except BaseException:
                    status.update(status='interrupted', exit_code=130)
                    write_json(status_path, status)
                    raise
            if code == 0:
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                duration = payload['arguments']['epochs']
                duration = payload['arguments']['steps'] if duration is None else duration
                if payload['iteration'] != duration:
                    raise RuntimeError(f'Incomplete stage checkpoint: {checkpoint}')
                del payload
            status.update(status='complete' if code == 0 else 'failed', exit_code=code)
            write_json(status_path, status)
            if code:
                raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--study', type=Path, required=True)
    prepare_parser.add_argument('--data-root', type=Path, required=True)
    prepare_parser.add_argument('--python', default=sys.executable)
    prepare_parser.add_argument('--backbone', choices=('dit', 'pixeldit'), default='pixeldit')
    prepare_parser.add_argument('--data-parallel', action='store_true')
    prepare_parser.add_argument('--reference', type=Path, help='Hudson submit_compact_dit manifest or run.training_command JSON')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--manifest', type=Path, required=True)
    run_parser.add_argument('--dry-run', action='store_true')
    run_parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    prepare(args) if args.action == 'prepare' else run(args)


if __name__ == '__main__':
    main()
