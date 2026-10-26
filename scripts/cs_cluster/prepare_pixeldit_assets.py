#!/usr/bin/env python3
"""Download/verify the existing baseline's FID weights; does not alter FID."""
import hashlib
from pathlib import Path
import sys
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fid import ASSETS, WEIGHTS

URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def main():
    expected = (ASSETS / 'inception.sha256').read_text().split()[0]
    if not WEIGHTS.exists():
        temporary = WEIGHTS.with_suffix('.download')
        print(f'Downloading {URL}', flush=True)
        urlretrieve(URL, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != expected:
            temporary.unlink()
            raise RuntimeError('Downloaded FID weight checksum mismatch')
        temporary.replace(WEIGHTS)
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError('Existing FID weight checksum mismatch')
    print(f'Verified {WEIGHTS}: {digest}')


if __name__ == '__main__':
    main()
