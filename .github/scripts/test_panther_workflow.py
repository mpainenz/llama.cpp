"""Check the workflow's admission expression against trusted/untrusted events."""
import itertools
import unittest
from pathlib import Path

import yaml


class PantherRoutingTests(unittest.TestCase):
    def test_linux_only_trial_does_not_start_windows(self):
        workflow = Path(__file__).resolve().parents[1] / "workflows/build-tensorrelay.yml"
        job = yaml.safe_load(workflow.read_text())["jobs"]["windows-cuda-x64"]
        self.assertEqual(job["if"], "${{ !inputs.local_linux_trial }}")

    def test_only_owner_branch_builds_use_panther(self):
        workflow = Path(__file__).resolve().parents[1] / "workflows/build-tensorrelay.yml"
        job = yaml.safe_load(workflow.read_text())["jobs"]["windows-cuda-x64"]
        expression = job["runs-on"].removeprefix("${{ ").removesuffix(" }}")
        expression = expression.replace("&&", "and").replace("||", "or")
        for repo, actor, event, ref in itertools.product(
            ["mpainenz/llama.cpp", "other/fork"],
            ["mpainenz", "other"],
            ["push", "workflow_dispatch", "pull_request", "schedule"],
            ["branch", "tag"],
        ):
            with self.subTest(repo=repo, actor=actor, event=event, ref=ref):
                case = expression
                for key, value in dict(repository=repo, actor=actor, event_name=event, ref_type=ref).items():
                    case = case.replace("github." + key, repr(value))
                actual = eval(case, {"__builtins__": {}}, {})
                trusted = (
                    repo == "mpainenz/llama.cpp"
                    and actor == "mpainenz"
                    and event in ("push", "workflow_dispatch")
                    and ref == "branch"
                )
                self.assertEqual(actual, "panther-windows-llama" if trusted else "windows-2022")


if __name__ == "__main__":
    unittest.main()
