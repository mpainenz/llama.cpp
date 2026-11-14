"""Exercise cache admission against real Git histories and CMake metadata."""
import importlib.util
import os
import shutil
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
            f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\nCMAKE_GENERATOR:INTERNAL=Ninja\nLLAMA_BUILD_TESTS:BOOL=OFF\n")

    def selected(self):
        return cache.select_cache(self.repo, self.root)

    def test_ci_only_commit_reuses_existing_source_cache(self):
        self.commit(".github/workflows/build.yml", "changed CI")
        self.assertEqual(self.selected(), self.old)

    def test_source_change_requires_new_cache(self):
        self.commit("src/main.cpp", "changed code")
        self.assertNotEqual(self.selected(), self.old)

    def test_executor_only_change_reuses_backend_objects(self):
        self.commit("tools/tensorrelay-stage-executor/main.cpp", "changed executor")
        self.assertEqual(self.selected(), self.old)

    def test_completed_cache_is_preferred_to_newer_partial_build(self):
        (self.old / "bin").mkdir()
        (self.old / "bin/tensorrelay_stage_executor.dll").touch()
        self.commit(".github/workflows/build.yml", "changed CI")
        partial = self.root / ("fork-" + self.git("rev-parse", "HEAD"))
        partial.mkdir()
        shutil.copyfile(self.old / "CMakeCache.txt", partial / "CMakeCache.txt")
        self.assertEqual(self.selected(), self.old)

    def test_test_changes_are_not_ignored_when_tests_are_enabled(self):
        config = self.old / "CMakeCache.txt"
        config.write_text(config.read_text().replace("LLAMA_BUILD_TESTS:BOOL=OFF", "LLAMA_BUILD_TESTS:BOOL=ON"))
        self.commit("tests/test.cpp", "changed test")
        self.assertNotEqual(self.selected(), self.old)

    @unittest.skipIf(os.name == "nt", "Native workflow builds the real MSVC target after loading its environment")
    def test_cmake_rebuilds_executor_without_recompiling_backend(self):
        self.commit("src/backend.cpp", "int backend() { return 0; }\n")
        self.commit("tools/tensorrelay-stage-executor/main.cpp", '#include <cstdio>\nint backend(); int main() { puts("first"); return backend(); }\n')
        self.commit("CMakeLists.txt", '''cmake_minimum_required(VERSION 3.16)
project(cache_check CXX)
option(LLAMA_BUILD_TESTS "Tests" OFF)
add_library(backend STATIC src/backend.cpp)
add_executable(tensorrelay_stage_executor tools/tensorrelay-stage-executor/main.cpp)
target_link_libraries(tensorrelay_stage_executor PRIVATE backend)
set_target_properties(tensorrelay_stage_executor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" SUFFIX ".dll")
''')
        build = self.root / ("fork-" + self.git("rev-parse", "HEAD"))

        def compile_target():
            subprocess.run(["cmake", "-S", str(self.repo), "-B", str(build), "-G", "Ninja"], check=True, capture_output=True)
            subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True)

        compile_target()
        binary = build / "bin/tensorrelay_stage_executor.dll"
        self.assertEqual(subprocess.check_output([str(binary)], text=True).strip(), "first")
        backend_time = (build / "libbackend.a").stat().st_mtime_ns
        self.commit("tools/tensorrelay-stage-executor/main.cpp", '#include <cstdio>\nint backend(); int main() { puts("second"); return backend(); }\n')
        self.assertEqual(self.selected(), build)
        compile_target()
        self.assertEqual(subprocess.check_output([str(binary)], text=True).strip(), "second")
        self.assertEqual((build / "libbackend.a").stat().st_mtime_ns, backend_time)

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
