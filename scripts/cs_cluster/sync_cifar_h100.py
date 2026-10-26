#!/usr/bin/env python3
"""Copy H100 previews while running; verify every final artifact after completion.

Usage: python scripts/cs_cluster/sync_cifar_h100.py path/to/deployment.json
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('deployment', type=Path)
parser.add_argument('--interval', type=int, default=300)
args = parser.parse_args()
deployment = json.loads(args.deployment.read_text())
remote_log = deployment.get('remote_log', deployment['remote_source'] + '/scheduler.log')
destination = Path(deployment['local_output'])
destination.mkdir(parents=True, exist_ok=True)
status_path = destination / 'transfer_status.json'

def record(state, **extra):
    payload = dict(status=state, job_id=deployment['job_id'],
                   updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(status_path)
    print(json.dumps(payload), flush=True)

ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'cs-cluster']
while True:
    try:
        remote = deployment['remote_output']
        # Deployment paths are generated locally and contain no shell metacharacters.
        import shlex
        response = subprocess.run(ssh + ['cat ' + shlex.quote(remote + '/job_status.json')],
                                  capture_output=True, text=True, timeout=90)
        if response.returncode:
            record('waiting_for_remote_status', detail=response.stderr[-1000:])
        else:
            state = json.loads(response.stdout)
            if state['status'] == 'running':
                subprocess.run(['rsync', '-a', '--delay-updates', '--exclude=*.tmp', '-e',
                                'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2',
                                'cs-cluster:' + remote + '/', str(destination) + '/'],
                               check=True, timeout=1800)
                subprocess.run(['rsync', '-a', '--delay-updates', '-e',
                                'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                'cs-cluster:' + remote_log,
                                str(destination / 'scheduler.log')], check=True, timeout=120)
                record('partial_synced', remote_status=state,
                       note='Run is incomplete; final artifact hashes are not yet available.')
            if state['status'] in ('complete', 'failed'):
                subprocess.run(['rsync', '-a', '--checksum', '-e',
                                'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2',
                                'cs-cluster:' + remote + '/', str(destination) + '/'],
                               check=True, timeout=1800)
                subprocess.run(['rsync', '-a', '-e',
                                'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                'cs-cluster:' + remote_log,
                                str(destination / 'scheduler.log')], check=True, timeout=120)
                if state['status'] == 'failed':
                    record('remote_failed_logs_copied', remote_status=state)
                    break
                manifest = json.loads((destination / 'artifact_manifest.json').read_text())
                for name, expected in manifest.items():
                    assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected, name
                record('complete_verified', verified_files=len(manifest), remote_status=state)
                break
    except Exception as error:
        record('retrying', detail=str(error))
    time.sleep(args.interval)
