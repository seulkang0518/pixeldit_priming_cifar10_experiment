#!/usr/bin/env python3
"""Run one CIFAR trainer in a dedicated GPU allocation with checkpoint retention."""
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

from run_cifar_pair import preserve_best_fid, set_option, training_command, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    record = manifest['run']
    threads = manifest.get('cpu_threads', 4)
    command = training_command(record)
    set_option(command, '--threads', threads)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output = Path(record['output'])
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), DRIFT_COMPILE='0', PYTHONUNBUFFERED='1')
    child = refresh = None

    def status(state, code=0):
        write_json(output / 'job_status.json', dict(stage='posttraining', status=state,
                   exit_code=code, job_id=os.environ.get('LSB_JOBID'), host=os.uname().nodename,
                   cpu_threads=threads, concurrent_runs_per_gpu=1,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   training_pid=child.pid if child is not None else None,
                   resumed='--resume' in command, manifest=str(args.manifest.resolve()),
                   updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        status('interrupted', 128 + signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (output / '.paired_training.lock').open('a') as lock, (output / 'run.log').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('gpu_smoke_check')
            child = subprocess.Popen(record['smoke_command'], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait():
                raise RuntimeError('GPU smoke check failed')
            log.write(shlex.join(command) + '\n')
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            status('running')
            last_refresh = None
            while child.poll() is None:
                try:
                    preserve_best_fid(record)
                except Exception as error:
                    print(f'Best-FID retention pending: {error}', flush=True)
                metrics = output / 'fid_metrics.json'
                stamp = metrics.stat().st_mtime_ns if metrics.exists() else None
                if (manifest.get('refresh_command') and stamp is not None and stamp != last_refresh
                        and (refresh is None or refresh.poll() is not None)):
                    refresh = subprocess.Popen(manifest['refresh_command'], env=env)
                    last_refresh = stamp
                time.sleep(5)
            if child.returncode:
                status('failed', child.returncode)
                raise SystemExit(child.returncode)
            preserve_best_fid(record)
            status('training_complete')
            subprocess.run(record['finalize_command'], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            status('complete')
        except Exception:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            status('failed', 1)
            raise
        finally:
            if refresh is not None:
                refresh.wait(timeout=300)


if __name__ == '__main__':
    main()
