"""Output placement for new runs; existing paths and explicit absolute overrides survive."""
import os
from pathlib import Path


# Standalone experiment: new runs belong to THIS copy, never Hudson's checkout.
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
LEGACY_RESULTS_ROOT = Path('/home/zonghaoc/mmd_flow_hudson_new/results')


def new_run_path(path):
    """Place new relative run names under this standalone experiment's results.

    Preserve existing paths for recovery/history. Absolute paths outside the old
    results root are explicit overrides (including temporary test directories).
    An explicit MMD_RESULTS_ROOT or absolute destination can select shared storage.
    """
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        return path
    root = Path(os.environ.get('MMD_RESULTS_ROOT', str(DEFAULT_RESULTS_ROOT))).expanduser()
    if not root.is_absolute():
        raise ValueError('MMD_RESULTS_ROOT must be an absolute path')
    if path.is_absolute():
        try:
            relative = path.relative_to(LEGACY_RESULTS_ROOT)
        except ValueError:
            return path
    else:
        relative = path.relative_to('results') if path.parts and path.parts[0] == 'results' else path
    if '..' in relative.parts:
        raise ValueError('Run names cannot contain parent-directory traversal; use an explicit absolute path')
    return root / relative


if __name__ == '__main__':
    import sys
    print(new_run_path(sys.argv[1]))
