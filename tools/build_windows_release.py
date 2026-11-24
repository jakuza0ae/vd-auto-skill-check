#!/usr/bin/env python3
"""Package the Windows x64 worker, Python and application into a portable ZIP."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = '3.14.10'
PYTHON_SERIES = '3.14'
PYTHON_ABI = '314'
PYTHON_SHA256 = '90905e64a840557d8c1bc4f5b74db3d7ee85e3ffb217611cac1c7813acea4159'
WORKER_PATH = Path('build/windows-capture/vd-windows-capture.exe')
ARCHIVE_NAME = 'vd-windows-x64.zip'
RUNTIME_PACKAGES = ('numpy', 'opencv-python-headless')


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install_python(runtime, staging, python_zip):
    if python_zip is None:
        name = f'python-{PYTHON_VERSION}-embed-amd64.zip'
        url = f'https://www.python.org/ftp/python/{PYTHON_VERSION}/{name}'
        python_zip = staging / name
        with urlopen(url, timeout=45) as response, python_zip.open('wb') as destination:
            shutil.copyfileobj(response, destination)
    if file_sha256(python_zip) != PYTHON_SHA256:
        raise ValueError(f'CPython {PYTHON_VERSION} x64 SHA-256 mismatch: {python_zip}')
    with ZipFile(python_zip) as archive:
        archive.extractall(runtime)
    # Restrict imports to the shipped interpreter, application and wheel contents.
    (runtime / f'python{PYTHON_ABI}._pth').write_text(
        f'python{PYTHON_ABI}.zip\n.\n..\nsite-packages\n', encoding='ascii')


def install_wheels(runtime, wheelhouse):
    requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8')
    pins = []
    for package in RUNTIME_PACKAGES:
        pattern = rf'^{re.escape(package)}==[A-Za-z0-9.]+[ \t]*$'
        matches = re.findall(pattern, requirements, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f'Expected one exact {package} pin in requirements.txt')
        pins.append(matches[0].strip())

    index = (['--no-index', '--find-links', str(wheelhouse.resolve())] if wheelhouse
             else ['--index-url', 'https://pypi.org/simple'])
    subprocess.run([
        sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check',
        'install', '--no-deps', '--no-compile', '--only-binary=:all:',
        '--platform', 'win_amd64', '--python-version', PYTHON_SERIES,
        '--implementation', 'cp', '--abi', f'cp{PYTHON_ABI}',
        '--target', str(runtime / 'site-packages'), *index, *pins,
    ], check=True)


def check_worker(worker):
    with worker.open('rb') as stream:
        header = stream.read(64)
        if len(header) == 64 and header[:2] == b'MZ':
            stream.seek(struct.unpack_from('<I', header, 60)[0])
            if stream.read(6) == b'PE\0\0\x64\x86':
                return
    raise ValueError(f'Expected a Windows x64 PE executable: {worker}')


def copy_application(bundle, worker):
    files = [
        ROOT / 'run.py', ROOT / 'run-vulkan.py', ROOT / 'vd.cmd', ROOT / 'README.md',
        ROOT / 'native/windows_capture.cpp', *sorted((ROOT / 'vd').glob('*.py')),
        *sorted((ROOT / 'assets').glob('space_template*.png')),
        *sorted(path for path in ROOT.glob('LICENSE*') if path.is_file()),
    ]
    for source in files:
        destination = bundle / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    destination = bundle / WORKER_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(worker, destination)


def build(output, worker, python_zip=None, wheelhouse=None):
    check_worker(worker)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vd-release-', dir=output.parent) as directory:
        staging = Path(directory)
        bundle = staging / 'VD-Windows-x64'
        copy_application(bundle, worker)
        install_python(bundle / 'python', staging, python_zip)
        install_wheels(bundle / 'python', wheelhouse)
        ready = staging / ARCHIVE_NAME
        with ZipFile(ready, mode='w', compression=ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle).as_posix())
        ready.replace(output)
    print(f'{output}  SHA-256 {file_sha256(output)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/releases' / ARCHIVE_NAME)
    parser.add_argument('--worker', type=Path, default=ROOT / WORKER_PATH)
    parser.add_argument('--python-zip', type=Path,
                        help=f'use a local official CPython {PYTHON_VERSION} x64 embeddable ZIP')
    parser.add_argument('--wheelhouse', type=Path, help='install Windows wheels from this directory')
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.worker, args.python_zip, args.wheelhouse)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Windows release build failed: {exc}\n')


if __name__ == '__main__':
    main()
