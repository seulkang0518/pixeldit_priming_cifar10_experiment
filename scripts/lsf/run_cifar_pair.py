#!/usr/bin/env python3
"""Run two independent CIFAR experiments inside one shared-compute GPU allocation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def set_option(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def training_command(record):
    command = list(record['training_command'])
    for flag in ('--resume', '--fork-checkpoint'):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    set_option(command, '--threads', 2)
    checkpoint = Path(record['output']) / 'checkpoint_latest.pt'
    if checkpoint.exists():
        set_option(command, '--resume', checkpoint)
    elif (Path(record['output']) / 'config.json').exists():
        raise RuntimeError(f'Existing config without recovery checkpoint: {record["output"]}')
    return command


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def preserve_best_fid(record):
    """Keep one immutable recovery checkpoint matching the best completed FID."""
    if not record.get('preserve_best_fid'):
        return
    output = Path(record['output'])
    metrics = output / 'fid_metrics.json'
    if not metrics.exists():
        return
    rows = json.loads(metrics.read_text())
    if not rows:
        return
    latest = rows[-1]
    archive = output / 'checkpoints' / 'best_fid'
    provenance = archive / 'metrics.json'
    if provenance.exists() and json.loads(provenance.read_text())['fid'] <= latest['fid']:
        return
    checkpoint = output / 'checkpoint_latest.pt'
    if not checkpoint.exists():
        return
    archive.mkdir(parents=True, exist_ok=True)
    candidate = archive / 'candidate.pt'
    try:
        os.link(checkpoint, candidate)
        import torch
        saved = torch.load(candidate, map_location='cpu', mmap=True, weights_only=False)
        update = saved['update_count']
        del saved
        if update != latest['iteration']:
            return  # FID is written before the corresponding checkpoint.
        candidate.replace(archive / 'checkpoint_latest.pt')
        write_json(provenance, dict(latest, checkpoint=str(archive / 'checkpoint_latest.pt')))
    finally:
        candidate.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest['runs']
    if len(records) != 2 or len({r['output'] for r in records}) != 2:
        parser.error('Exactly two different run directories are required')
    commands = [training_command(record) for record in records]
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', DRIFT_COMPILE='0')
    locks, streams, children = [], [], []
    refresh = None
    refresh_signature = None

    def status(record, state, code=0, **extra):
        path = Path(record['output']) / 'job_status.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update(stage='posttraining', status=state, exit_code=code,
                        job_id=os.environ.get('LSB_JOBID'), host=os.uname().nodename,
                        shared_gpu_manifest=str(args.manifest.resolve()),
                        cpu_threads=2, concurrent_runs_per_gpu=2,
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        target_updates=250000,
                        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        write_json(path, previous)

    def terminate(signum, frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for record in records:
            status(record, 'interrupted', 128 + signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for record in records:
            output = Path(record['output'])
            output.mkdir(parents=True, exist_ok=True)
            lock = (output / '.paired_training.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        # Run the existing full-width checks sequentially, then launch both trainers.
        for record in records:
            status(record, 'gpu_smoke_check')
            with (Path(record['output']) / 'paired_worker.log').open('a', buffering=1) as stream:
                subprocess.run(record['smoke_command'], env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        for record, command in zip(records, commands):
            stream = (Path(record['output']) / 'run.log').open('a', buffering=1)
            streams.append(stream)
            stream.write(shlex.join(command) + '\n')
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(child)
            status(record, 'running', training_pid=child.pid,
                   resumed='--resume' in command)
        remaining = set(range(len(children)))
        failed = False
        while remaining:
            for record in records:
                try:
                    preserve_best_fid(record)
                except Exception as error:
                    print(f'Best-FID retention pending: {error}', flush=True)
            if manifest.get('refresh_command'):
                signature = tuple((Path(r['output']) / 'fid_metrics.json').stat().st_mtime_ns
                                  if (Path(r['output']) / 'fid_metrics.json').exists() else 0
                                  for r in records)
                if any(signature) and signature != refresh_signature and (refresh is None or refresh.poll() is not None):
                    refresh_signature = signature
                    refresh = subprocess.Popen(manifest['refresh_command'], env=env)
            for index in list(remaining):
                code = children[index].poll()
                if code is None:
                    continue
                remaining.remove(index)
                record = records[index]
                if code:
                    status(record, 'failed', code)
                    failed = True
                    continue
                status(record, 'training_complete')
                with (Path(record['output']) / 'paired_worker.log').open('a', buffering=1) as stream:
                    result = subprocess.run(record['finalize_command'], env=env, stdout=stream,
                                            stderr=subprocess.STDOUT)
                status(record, 'complete' if result.returncode == 0 else 'finalization_failed',
                       result.returncode)
                failed |= result.returncode != 0
            if remaining:
                time.sleep(5)
        if failed:
            raise SystemExit(1)
    except Exception:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for record in records:
            status(record, 'worker_failed', 1)
        raise
    finally:
        if refresh is not None:
            refresh.wait(timeout=300)
        for stream in streams + locks:
            stream.close()


if __name__ == '__main__':
    main()
