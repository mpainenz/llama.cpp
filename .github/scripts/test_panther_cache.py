"""Exercise cache admission against real Git histories and CMake metadata."""
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("cache", Path(__file__).with_name("select-panther-cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class PantherCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.root = Path(self.tmp.name) / "cache"
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.commit("src/main.cpp", "first")
        self.old = self.root / ("fork-" + self.git("rev-parse", "HEAD"))
        self.old.mkdir(parents=True)
        self.configure(self.repo)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def commit(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git("add", ".")
        self.git("commit", "-qm", "test")

    def configure(self, home):
        (self.old / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\nCMAKE_GENERATOR:INTERNAL=Ninja\n")

    def selected(self):
        return cache.select_cache(self.repo, self.root)

    def test_ci_only_commit_reuses_existing_source_cache(self):
        self.commit(".github/workflows/build.yml", "changed CI")
        self.assertEqual(self.selected(), self.old)

    def test_source_change_requires_new_cache(self):
        self.commit("src/main.cpp", "changed code")
        self.assertNotEqual(self.selected(), self.old)

    def test_cmake_change_requires_new_cache(self):
        self.commit("CMakeLists.txt", "changed configuration")
        self.assertNotEqual(self.selected(), self.old)

    def test_other_workspace_cannot_be_reused(self):
        self.configure(self.repo.parent / "different")
        self.commit(".github/workflows/build.yml", "changed CI")
        self.assertNotEqual(self.selected(), self.old)

    def test_missing_configuration_cannot_be_reused(self):
        (self.old / "CMakeCache.txt").unlink()
        self.commit(".github/workflows/build.yml", "changed CI")
        self.assertNotEqual(self.selected(), self.old)


if __name__ == "__main__":
    unittest.main()
