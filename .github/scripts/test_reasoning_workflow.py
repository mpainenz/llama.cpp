"""The reasoning check must use its CMake-linked target, not a standalone link."""
import unittest
from pathlib import Path
import yaml

class ReasoningWorkflowTests(unittest.TestCase):
    def test_release_builds_and_runs_registered_cmake_test(self):
        workflow = Path(__file__).resolve().parents[1] / 'workflows/build-tensorrelay.yml'
        steps = yaml.safe_load(workflow.read_text())['jobs']['linux-vulkan-x64']['steps']
        build = next(s['run'] for s in steps if s.get('name') == 'Build stage executor')
        self.assertGreater(build.index('-DLLAMA_BUILD_TESTS=ON'), build.index('${{ env.CMAKE_COMMON }}'))
        self.assertIn('--target tensorrelay_stage_executor test-tensorrelay-chat-options', build)
        check = next(s['run'] for s in steps if s.get('name') == 'Test reasoning controls with linked common libraries')
        self.assertIn("ctest --test-dir build -R '^test-tensorrelay-chat-options$' --output-on-failure", check)
        self.assertNotIn('c++ -std=', '\n'.join(s.get('run', '') for s in steps))

if __name__ == '__main__':
    unittest.main()
