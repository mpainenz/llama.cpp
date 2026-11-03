#!/usr/bin/env python3
"""Regression coverage for the packaged library runtime gate."""
import importlib.util
from pathlib import Path
import unittest
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

if __name__ == '__main__':
    unittest.main()
