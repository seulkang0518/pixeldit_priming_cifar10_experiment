#!/usr/bin/env python3
"""Archive a completed training checkpoint and refresh the comparison notebook."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--expected-update', type=int)
    parser.add_argument('--notebook', type=Path, required=True)
    args = parser.parse_args()
    if (args.run is None) != (args.expected_update is None):
        parser.error('--run and --expected-update must be supplied together')
    if args.run:
        import torch
        checkpoint = args.run / 'checkpoint_latest.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved['update_count'] != args.expected_update:
            raise ValueError(f"Expected update {args.expected_update}, got {saved['update_count']}")
        del saved
        archive = args.run / 'checkpoints' / f'step_{args.expected_update:06d}'
        archive.mkdir(parents=True, exist_ok=True)
        for source in [checkpoint, args.run / 'metrics.csv', *args.run.glob('*.json')]:
            if source.is_file():
                temporary = archive / (source.name + '.tmp')
                shutil.copy2(source, temporary)
                temporary.replace(archive / source.name)
        print(f'Preserved checkpoint, optimizer/RNG state, and metrics: {archive}', flush=True)

    import nbformat
    from nbclient import NotebookClient
    notebook = args.notebook.resolve()
    # Multiple completed training jobs may request a refresh at the same time.
    with notebook.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content = nbformat.read(notebook, as_version=4)
        client = NotebookClient(content, timeout=300, kernel_name='python3',
                                resources={'metadata': {'path': str(notebook.parent.parent)}})
        client.execute()
        fd, name = tempfile.mkstemp(prefix=notebook.stem, suffix='.ipynb', dir=notebook.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                nbformat.write(content, stream)
            Path(name).replace(notebook)
        finally:
            Path(name).unlink(missing_ok=True)
        print(f'Refreshed {notebook}', flush=True)


if __name__ == '__main__':
    main()
