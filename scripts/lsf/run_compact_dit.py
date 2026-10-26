"""Run the recorded DiT smoke check, reconstruction, and Riesz post-training."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

from run_cifar_pair import set_option, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    study = Path(manifest['study'])
    if args.dry_run:
        print(shlex.join(manifest['smoke_command']))
        for record in manifest['stages']:
            print(shlex.join(record['command']))
        return
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               DRIFT_COMPILE='0', PYTHONUNBUFFERED='1')
    child = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (study / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (study / 'smoke.log').open('a', buffering=1) as stream:
            child = subprocess.Popen(manifest['smoke_command'], env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait():
                raise RuntimeError(f'GPU smoke check failed; inspect {study / "smoke.log"}')
        for record in manifest['stages']:
            output = Path(record['output'])
            status_path = output / 'job_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status') == 'complete':
                continue
            command = list(record['command'])
            if (output / 'checkpoint_latest.pt').exists():
                set_option(command, '--resume', output / 'checkpoint_latest.pt')
            status = dict(stage=record['stage'], status='running', job_id=os.environ.get('LSB_JOBID'),
                          host=os.uname().nodename, command=command,
                          updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            with (output / 'run.log').open('a', buffering=1) as stream:
                stream.write(shlex.join(command) + '\n')
                child = subprocess.Popen(command, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                status['training_pid'] = child.pid
                write_json(status_path, status)
                try:
                    code = child.wait()
                except BaseException:
                    status.update(status='interrupted', exit_code=130)
                    write_json(status_path, status)
                    raise
            status.update(status='complete' if code == 0 else 'failed', exit_code=code,
                          updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(status_path, status)
            if code:
                raise SystemExit(code)


if __name__ == '__main__':
    main()
