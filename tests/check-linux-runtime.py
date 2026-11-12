#!/usr/bin/env python3
"""Reject packaged ELF libraries that exceed the supported Linux ABI baseline."""
import pathlib
import re
import subprocess
import sys

# Existing Ubuntu 22.04 release requirements, not the builder's own versions.
LIMITS = {'GLIBC': (2, 34), 'GLIBCXX': (3, 4, 30), 'CXXABI': (1, 3, 13)}


def violations(text):
    # Only inspect dependencies (version needs), not versions this ELF defines.
    needs = text.partition("Version needs section")[2]
    errors = []
    for family, value in re.findall(r'Name: (GLIBCXX|GLIBC|CXXABI)_([0-9.]+)', needs):
        if tuple(map(int, value.split('.'))) > LIMITS[family]:
            errors.append(f'{family}_{value} exceeds {family}_{".".join(map(str, LIMITS[family]))}')
    return sorted(set(errors))


def missing_dependencies(text, packaged):
    needed = re.findall(r'\(NEEDED\).*\[([^]]+)\]', text)
    return sorted({name for name in needed if name.startswith(('libllama', 'libggml', 'libtensorrelay')) and name not in packaged})


def main():
    libraries = sorted(pathlib.Path(sys.argv[1]).glob('*.so*'))
    if not libraries:
        raise SystemExit('No packaged libraries found; ABI check cannot pass')
    failed = False
    packaged = {library.name for library in libraries}
    for library in libraries:
        text = subprocess.check_output(['readelf', '--version-info', str(library)], text=True)
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        errors = violations(text) + [f'missing packaged dependency: {name}' for name in missing_dependencies(dynamic, packaged)]
        for error in errors:
            print(f'{library.name}: {error}', file=sys.stderr)
            failed = True
    if failed:
        raise SystemExit(1)
    print(f'Linux runtime baseline verified for {len(libraries)} libraries')


if __name__ == '__main__':
    main()
