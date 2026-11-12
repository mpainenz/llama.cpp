"""Reuse unchanged backend builds; CMake recompiles changed executor sources."""
import subprocess
import sys
from pathlib import Path


def select_cache(repo, cache_root):
    repo, cache_root = Path(repo).resolve(), Path(cache_root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    revisions = git("rev-list", "--first-parent", "--max-count=8", "HEAD").splitlines()
    current = cache_root / ("fork-" + revisions[0])
    compatible = []
    for revision in revisions:
        candidate = cache_root / ("fork-" + revision)
        config = candidate / "CMakeCache.txt"
        if not config.is_file():
            continue
        fields = dict(line.split("=", 1) for line in config.read_text().splitlines()
                      if line.startswith(("CMAKE_HOME_DIRECTORY:INTERNAL=", "CMAKE_GENERATOR:INTERNAL=", "LLAMA_BUILD_TESTS:BOOL=")))
        home = fields.get("CMAKE_HOME_DIRECTORY:INTERNAL")
        if not home or Path(home).resolve() != repo or fields.get("CMAKE_GENERATOR:INTERNAL") != "Ninja":
            continue
        # CMake rebuilds the executor incrementally. Reuse backend objects only
        # when their sources, common code and root configuration are identical.
        exclusions = [":(exclude).github", ":(exclude)tools/tensorrelay-stage-executor"]
        if fields.get("LLAMA_BUILD_TESTS:BOOL") == "OFF":
            exclusions.append(":(exclude)tests")
        difference = subprocess.run([
            "git", "-C", str(repo), "diff", "--quiet", revision, "HEAD", "--", ".", *exclusions,
        ], check=False)
        if difference.returncode == 0:
            compatible.append(candidate)
        elif difference.returncode != 1:
            raise RuntimeError("Cannot verify cached source tree")
    # A completed compatible build saves more work than a newer partial one.
    for candidate in compatible:
        if (candidate / "bin" / "tensorrelay_stage_executor.dll").is_file():
            return candidate
    return compatible[0] if compatible else current


if __name__ == "__main__":
    print(select_cache(sys.argv[1], sys.argv[2]))
