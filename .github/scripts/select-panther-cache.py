"""Reuse a Panther build for CI-only commits; CMake still runs on every build."""
import subprocess
import sys
from pathlib import Path


def select_cache(repo, cache_root):
    repo, cache_root = Path(repo).resolve(), Path(cache_root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    revisions = git("rev-list", "--first-parent", "--max-count=8", "HEAD").splitlines()
    current = cache_root / ("fork-" + revisions[0])
    if current.exists():
        return current
    for revision in revisions[1:]:
        candidate = cache_root / ("fork-" + revision)
        config = candidate / "CMakeCache.txt"
        if not config.is_file():
            continue
        fields = dict(line.split("=", 1) for line in config.read_text().splitlines()
                      if line.startswith(("CMAKE_HOME_DIRECTORY:INTERNAL=", "CMAKE_GENERATOR:INTERNAL=")))
        home = fields.get("CMAKE_HOME_DIRECTORY:INTERNAL")
        if not home or Path(home).resolve() != repo or fields.get("CMAKE_GENERATOR:INTERNAL") != "Ninja":
            continue
        # Include every tracked input except CI scripts. Never reuse a changed
        # source, header, test, submodule pointer, or CMake configuration tree.
        difference = subprocess.run([
            "git", "-C", str(repo), "diff", "--quiet", revision, "HEAD", "--", ".", ":(exclude).github",
        ], check=False)
        if difference.returncode == 0:
            return candidate
        if difference.returncode != 1:
            raise RuntimeError("Cannot verify cached source tree")
    return current


if __name__ == "__main__":
    print(select_cache(sys.argv[1], sys.argv[2]))
