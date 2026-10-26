#!/usr/bin/env python3
"""Run a tuned continuation, or restart post-training from reconstruction."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--stage', choices=('baseline', 'posttraining'), required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--restart-from-priming', type=Path,
                        help='Use reconstruction weights with fresh optimizer/RNG at update zero')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--objective', choices=('softmax-riesz', 'riesz', 'sinkhorn'),
                        help='Override the objective from the saved argument recipe')
    parser.add_argument('--no-field-rms-normalization', action='store_true')
    parser.add_argument('--sinkhorn-mode', choices=('wflow', 'ot'), default='wflow')
    parser.add_argument('--sinkhorn-iters', type=int, default=10)
    parser.add_argument('--sinkhorn-tolerance', type=float, default=1e-6)
    parser.add_argument('--sinkhorn-ot-epsilon', type=float, default=None)
    args = parser.parse_args()
    if args.restart_from_priming and args.stage != 'posttraining':
        parser.error('--restart-from-priming requires --stage posttraining')
    if not 0 <= args.min_learning_rate <= args.learning_rate or args.learning_rate <= 0:
        parser.error('Require 0 <= minimum learning rate <= positive peak learning rate')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.update(PYTHONUNBUFFERED='1', DRIFT_COMPILE='0')
    import torch
    record = json.loads((args.study / 'paused_runs.json').read_text())[args.stage]
    checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    options = dict(checkpoint['arguments'])
    del checkpoint
    output = args.study / args.stage
    options.update(output=str(output), steps=250000, learning_rate=args.learning_rate,
                   lr_schedule='cosine', min_learning_rate=args.min_learning_rate,
                   schedule_start_update=record['source_update'], warmup_steps=args.warmup_steps,
                   train_particles=128, random_flip=True, max_grad_norm=1., ema_decay=.9999,
                   crops_per_size=4, global_loss_weight=2., coarse_loss_weight=1.,
                   patch_loss_weight=1., checkpoint_every=1000, keep_checkpoint_every=25000,
                   resume=None, fork_checkpoint=record['checkpoint'])
    if args.objective is not None:
        options['objective'] = args.objective
    if options['objective'] == 'sinkhorn':
        options.update(sinkhorn_mode=args.sinkhorn_mode, sinkhorn_iters=args.sinkhorn_iters,
                       sinkhorn_tolerance=args.sinkhorn_tolerance,
                       sinkhorn_ot_epsilon=args.sinkhorn_ot_epsilon)
    if args.no_field_rms_normalization:
        if options['objective'] != 'softmax-riesz':
            parser.error('--no-field-rms-normalization requires softmax-riesz')
        options.update(no_field_rms_normalization=True, normalize_per_temperature=False)
    if args.restart_from_priming:
        # The preserved distributional checkpoint supplies arguments only.
        # main.py transfers reconstruction weights without optimizer or RNG state.
        options.update(init_checkpoint=str(args.restart_from_priming.resolve()),
                       fork_checkpoint=None, schedule_start_update=0)
    if (output / 'checkpoint_latest.pt').exists():
        options.update(resume=str(output / 'checkpoint_latest.pt'), fork_checkpoint=None)
    source = args.study / 'source'
    command = [sys.executable, '-u', str(source / 'main.py')]
    for key, value in options.items():
        flag = '--' + key.replace('_', '-')
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif key == 'normalize_per_temperature':
                command.append('--no-normalize-per-temperature')
        else:
            command += [flag, *map(str, value if isinstance(value, list) else [value])]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)

    def status(state, code=0):
        value = dict(stage=args.stage, status=state, exit_code=code,
                     job_id=os.environ.get('LSB_JOBID'), host=os.uname().nodename,
                     source_update=0 if args.restart_from_priming else record['source_update'],
                     initialization_checkpoint=options.get('init_checkpoint'),
                     target_updates=250000,
                     updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = output / 'job_status.json.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(output / 'job_status.json')

    status('running')
    try:
        with (output / 'run.log').open('a', buffering=1) as stream:
            stream.write(shlex.join(command) + '\n')
            subprocess.run([sys.executable, '-u', str(source / 'scripts/lsf/smoke_tuned_cifar.py'),
                            str(output / 'smoke.json'), str(args.learning_rate),
                            options['objective'],
                            'none' if args.no_field_rms_normalization else 'per_temperature',
                            args.sinkhorn_mode, str(args.sinkhorn_iters), str(args.sinkhorn_tolerance),
                            str(args.sinkhorn_ot_epsilon) if args.sinkhorn_ot_epsilon is not None else 'none'],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        saved = torch.load(output / 'checkpoint_latest.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['update_count'] == 250000
        del saved
        assert json.loads((output / 'fid_metrics.json').read_text())[-1]['iteration'] == 250000
        status('training_complete')
    except Exception:
        status('failed', 1)
        raise
    # Retention and notebook refresh use this job; no additional scheduler jobs.
    subprocess.run([sys.executable, str(source / 'scripts/lsf/finalize_cifar_comparison.py'),
                    '--run', str(output), '--expected-update', '250000',
                    '--notebook', str(args.notebook)], check=True)
    status('complete')


if __name__ == '__main__':
    main()
