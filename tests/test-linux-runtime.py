#!/usr/bin/env python3
"""Regression coverage for the packaged library runtime gate."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import subprocess
import sys
import shutil
spec = importlib.util.spec_from_file_location('runtime', Path(__file__).with_name('check-linux-runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

class RuntimeBaseline(unittest.TestCase):
    def test_previous_release(self):
        self.assertEqual(runtime.violations('Version needs section Name: GLIBC_2.34 Name: GLIBCXX_3.4.30 Name: CXXABI_1.3.13'), [])
    def test_new_toolchain_rejected(self):
        self.assertEqual(len(runtime.violations('Version needs section Name: GLIBC_2.38 Name: GLIBCXX_3.4.32')), 2)
    def test_definitions_not_requirements(self):
        self.assertEqual(runtime.violations('Version definition section Name: GLIBC_9.99\nVersion needs section Name: GLIBC_2.2.5'), [])
    def test_numeric_comparison(self):
        self.assertEqual(runtime.violations('Version needs section Name: GLIBC_2.9'), [])

class RuntimeClosure(unittest.TestCase):
    def test_common_library_must_be_bundled(self):
        text = ' 0x1 (NEEDED) Shared library: [libllama-common.so.0]'
        self.assertEqual(runtime.missing_dependencies(text, {'libllama.so.0'}), ['libllama-common.so.0'])
        self.assertEqual(runtime.missing_dependencies(text, {'libllama-common.so.0'}), [])

    def test_system_dependencies_remain_external(self):
        text = ' 0x1 (NEEDED) Shared library: [libc.so.6]'
        self.assertEqual(runtime.missing_dependencies(text, set()), [])

class PackagedDependencyIntegration(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc') and shutil.which('readelf'), 'ELF compiler required')
    def test_real_elf_missing_common_fails_then_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'common.c').write_text('int common(void) { return 1; }')
            (root / 'executor.c').write_text('extern int common(void); int executor(void) { return common(); }')
            subprocess.run(['cc', '-shared', '-fPIC', str(root / 'common.c'), '-Wl,-soname,libllama-common.so.0', '-o', str(root / 'libllama-common.so.0')], check=True)
            subprocess.run(['cc', '-shared', '-fPIC', str(root / 'executor.c'), '-L'+str(root), '-l:libllama-common.so.0', '-o', str(root / 'libtensorrelay_stage_executor.so')], check=True)
            command = [sys.executable, str(Path(__file__).with_name('check-linux-runtime.py')), str(root)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            (root / 'libllama-common.so.0').unlink()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing packaged dependency: libllama-common.so.0', result.stderr)

if __name__ == '__main__':
    unittest.main()
