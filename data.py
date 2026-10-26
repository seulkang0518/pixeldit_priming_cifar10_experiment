"""Checksum-verified MNIST and CIFAR-10 data loading."""
from pathlib import Path
import gzip
import hashlib
import pickle
import struct
import tarfile
import urllib.request

import numpy as np
import torch

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download_mnist(root: Path) -> Path:
    """Download all four canonical gzip files, atomically, with MD5 checks."""
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        path = raw / name
        if path.exists() and hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() == checksum:
            continue
        temporary = path.with_suffix(".gz.part")
        print(f"Downloading {name} to {path}", flush=True)
        try:
            urllib.request.urlretrieve(BASE_URL + name, temporary)
            if hashlib.md5(temporary.read_bytes(), usedforsecurity=False).hexdigest() != checksum:
                raise RuntimeError(f"MNIST checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return raw



def load_split(raw: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = "train" if train else "t10k"
    with gzip.open(raw / f"{prefix}-images-idx3-ubyte.gz", "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        data = stream.read()
    if magic != 2051 or (rows, columns) != (28, 28) or len(data) != count * 784:
        raise RuntimeError("Invalid MNIST image file")
    images = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()).reshape(count, 784)
    with gzip.open(raw / f"{prefix}-labels-idx1-ubyte.gz", "rb") as stream:
        magic, label_count = struct.unpack(">II", stream.read(8))
        data = stream.read()
    if magic != 2049 or label_count != count or len(data) != count:
        raise RuntimeError("Invalid MNIST label file")
    labels = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
    return images, labels



CIFAR10_FILENAME = 'cifar-10-binary.tar.gz'



CIFAR10_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'



CIFAR10_ARCHIVES = {CIFAR10_FILENAME: CIFAR10_MD5,
                    'cifar-10-python.tar.gz': 'c58f30108f718f92721af3b95e74349a'}



CIFAR10_URL = 'https://cave.cs.toronto.edu/kriz/' + CIFAR10_FILENAME



CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']



def download_cifar10(root: Path) -> Path:
    archive = root / 'CIFAR10' / 'raw' / CIFAR10_FILENAME
    candidates = [archive, archive.with_name('cifar-10-python.tar.gz'),
                  root / 'cifar-10-python.tar.gz']
    for candidate in candidates:
        if candidate.exists() and hashlib.md5(candidate.read_bytes(), usedforsecurity=False).hexdigest() == CIFAR10_ARCHIVES[candidate.name]:
            return candidate
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.gz.part')
    print(f'Downloading CIFAR-10 to {archive}', flush=True)
    try:
        urllib.request.urlretrieve(CIFAR10_URL, temporary)
        if hashlib.md5(temporary.read_bytes(), usedforsecurity=False).hexdigest() != CIFAR10_MD5:
            raise RuntimeError('CIFAR-10 checksum mismatch')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive



def decode_binary_batch(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    if not data or len(data) % 3073:
        raise RuntimeError('Invalid CIFAR-10 binary record length')
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3073)
    if np.any(records[:, 0] > 9):
        raise RuntimeError('Invalid CIFAR-10 class label')
    # Pixels are stored in channel-first R, G, B order.
    return torch.from_numpy(records[:, 1:].copy()), torch.from_numpy(records[:, 0].copy())



def load_cifar10_split(archive: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest() != CIFAR10_ARCHIVES.get(archive.name):
        raise RuntimeError('CIFAR-10 checksum mismatch')
    binary = archive.name == CIFAR10_FILENAME
    suffix = '.bin' if binary else ''
    prefix = 'cifar-10-batches-bin/' if binary else 'cifar-10-batches-py/'
    names = ([f'data_batch_{i}{suffix}' for i in range(1, 6)] if train else [f'test_batch{suffix}'])
    images, labels = [], []
    with tarfile.open(archive, 'r:gz') as stream:
        for name in names:
            member = stream.extractfile(prefix + name)
            if member is None:
                raise RuntimeError(f'Missing CIFAR-10 batch: {name}')
            if binary:
                x, y = decode_binary_batch(member.read())
            else:
                # Only unpickle after verifying the official archive's published checksum.
                record = pickle.load(member, encoding='bytes')
                data = record[b'data']
                labels_array = np.asarray(record[b'labels'], dtype=np.int64)
                if data.shape != (10000, 3072) or data.dtype != np.uint8:
                    raise RuntimeError(f'Invalid CIFAR-10 pixel data: {name}')
                if labels_array.shape != (10000,) or np.any((labels_array < 0) | (labels_array > 9)):
                    raise RuntimeError(f'Invalid CIFAR-10 labels: {name}')
                x, y = torch.from_numpy(data.copy()), torch.from_numpy(labels_array.copy())
            if len(x) != 10000:
                raise RuntimeError(f'Invalid CIFAR-10 batch size: {name}')
            images.append(x)
            labels.append(y)
    return torch.cat(images), torch.cat(labels)



def image_shape(dataset):
    if dataset == "mnist":
        return (1, 28, 28)
    if dataset == "cifar10":
        return (3, 32, 32)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_dataset(dataset, root):
    """Return the verified source and both splits using one dataset interface."""
    if dataset == "mnist":
        raw = download_mnist(root)
        loader = load_split
    elif dataset == "cifar10":
        raw = download_cifar10(root)
        loader = load_cifar10_split
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return raw, loader(raw, True), loader(raw, False)
