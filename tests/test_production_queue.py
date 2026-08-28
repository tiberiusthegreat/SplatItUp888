from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def powershell_environment() -> dict[str, str]:
    environment = os.environ.copy()
    module_path = environment.pop("PSMODULEPATH", None)
    if module_path:
        environment["PSModulePath"] = module_path
    return environment


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class ProductionSafetyTests(unittest.TestCase):
    def test_disk_gate_and_exclusive_lock_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = str(ROOT / "pipeline" / "production_safety.ps1").replace("'", "''")
            harness = root / "safety.ps1"
            test_root = str(root).replace("'", "''")
            harness.write_text(
                textwrap.dedent(
                    f"""
                    $ErrorActionPreference = "Stop"
                    . '{helper}'
                    try {{
                        Assert-SplatDiskCapacity -Path '{test_root}' -RequiredGB 1000000000 | Out-Null
                        throw "Disk gate unexpectedly passed"
                    }} catch {{
                        if ($_.Exception.Message -notlike "INSUFFICIENT_DISK_SPACE:*") {{ throw }}
                    }}
                    $lockPath = Join-Path '{test_root}' "exclusive.lock"
                    $first = Enter-SplatExclusiveLock -Path $lockPath -Metadata @{{ kind = "test" }}
                    try {{
                        try {{
                            Enter-SplatExclusiveLock -Path $lockPath -Metadata @{{ kind = "collision" }} | Out-Null
                            throw "Lock collision unexpectedly passed"
                        }} catch {{
                            if ($_.Exception.Message -notlike "SPLAT_LOCKED:*") {{ throw }}
                        }}
                    }} finally {{ Exit-SplatExclusiveLock $first }}
                    $second = Enter-SplatExclusiveLock -Path $lockPath -Metadata @{{ kind = "recovery" }}
                    Exit-SplatExclusiveLock $second
                    if (Test-Path -LiteralPath $lockPath) {{ throw "Released lock file remained" }}
                    "PRODUCTION_SAFETY_PASS"
                    """
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                capture_output=True,
                text=True,
                check=False,
                env=powershell_environment(),
            )
            self.assertEqual(0, result.returncode, result.stderr or result.stdout)
            self.assertIn("PRODUCTION_SAFETY_PASS", result.stdout)

    def test_core_runner_holds_global_and_per_run_locks(self) -> None:
        source = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        self.assertIn('Join-Path $OutputRoot ".splatitup.gpu.lock"', source)
        self.assertIn('Join-Path $RunRoot ".splatitup.run.lock"', source)
        self.assertIn("Assert-SplatDiskCapacity", source)
        self.assertIn("Exit-SplatExclusiveLock $RunLock", source)
        self.assertIn("Exit-SplatExclusiveLock $GlobalLock", source)


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class BatchQueueBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.pipeline = self.project / "pipeline"
        self.pipeline.mkdir(parents=True)
        shutil.copy2(ROOT / "pipeline" / "run_batch_queue.ps1", self.pipeline)
        shutil.copy2(ROOT / "pipeline" / "production_safety.ps1", self.pipeline)
        stub = textwrap.dedent(
            r"""
            [CmdletBinding()]
            param(
                [string]$VideoPath, [string]$RunName, [string]$SceneType,
                [string]$AutoPreset, [int]$SelectedFrames, [int]$TrainingSteps,
                [string]$Trainer, [int]$CandidateMultiplier,
                [double]$MaxCumulativeFlow, [int]$MaxLongSide,
                [int]$TrainingMaxResolution, [string]$OutputRoot,
                [string]$ConfigPath, [string]$ToStage,
                [switch]$NoAutoRotate, [switch]$NoBlender, [switch]$NoBrowser
            )
            $runRoot = Join-Path $OutputRoot $RunName
            New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
            "$RunName|$VideoPath" | Add-Content -LiteralPath $env:BATCH_TEST_LOG
            if ($VideoPath -like "*fail*") {
                $once = Join-Path $runRoot "failed-once.marker"
                if (-not (Test-Path -LiteralPath $once)) {
                    "failed" | Set-Content -LiteralPath $once
                    exit 1
                }
            }
            "complete" | Set-Content -LiteralPath (Join-Path $runRoot "complete.marker")
            exit 0
            """
        ).strip()
        (self.pipeline / "run_video_to_splat.ps1").write_text(stub, encoding="utf-8")
        (self.pipeline / "run_auto_video_to_splat.ps1").write_text(stub, encoding="utf-8")
        self.output = self.root / "runs"
        self.output.mkdir()
        self.config = self.project / "splatitup.local.psd1"
        self.config.write_text(
            f"@{{ OutputRoot = '{str(self.output).replace("'", "''")}'; Production = @{{ MinimumFreeSpaceGB = 0.01; PerQueuedJobReserveGB = 0.01; PlannedQueueSize = 2 }} }}",
            encoding="utf-8",
        )
        self.call_log = self.root / "calls.log"
        self.environment = powershell_environment()
        self.environment["BATCH_TEST_LOG"] = str(self.call_log)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_queue(self, videos: list[Path], queue: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        invocation = (
            f"& {quote(self.pipeline / 'run_batch_queue.ps1')} "
            f"-OutputRoot {quote(self.output)} -ConfigPath {quote(self.config)} "
            f"-QueuePath {quote(queue)} -CaptureType Object -Mode Preview -NoBlender"
        )
        if videos:
            video_array = "@(" + ",".join(quote(video) for video in videos) + ")"
            invocation += f" -VideoPaths {video_array}"
        if extra:
            invocation += " " + " ".join(extra)
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", invocation],
            capture_output=True,
            text=True,
            check=False,
            env=self.environment,
            timeout=30,
        )

    def make_video(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.write_bytes(payload)
        return path

    def test_queue_serializes_unique_source_bound_jobs(self) -> None:
        first = self.make_video("capture.mp4", b"first-video")
        second = self.make_video("capture-copy.mp4", b"second-video")
        queue = self.output / ".queues" / "two-jobs.json"
        result = self.run_queue([first, second], queue)
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)
        payload = json.loads(queue.read_text(encoding="utf-8-sig"))
        self.assertEqual("COMPLETE", payload["status"])
        self.assertEqual(["COMPLETE", "COMPLETE"], [job["status"] for job in payload["jobs"]])
        self.assertEqual(2, len(self.call_log.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(
            f"capture-{hashlib.sha256(b'first-video').hexdigest()[:8]}", payload["jobs"][0]["run_name"]
        )
        self.assertNotEqual(payload["jobs"][0]["run_name"], payload["jobs"][1]["run_name"])

    def test_failed_job_continues_and_resume_retries_it(self) -> None:
        failed = self.make_video("fail.mp4", b"fails-once")
        good = self.make_video("good.mp4", b"always-good")
        queue = self.output / ".queues" / "resume.json"
        first = self.run_queue([failed, good], queue)
        self.assertEqual(1, first.returncode, first.stderr or first.stdout)
        payload = json.loads(queue.read_text(encoding="utf-8-sig"))
        self.assertEqual(["FAILED", "COMPLETE"], [job["status"] for job in payload["jobs"]])

        resumed = self.run_queue([], queue, "-Resume", "-RetryFailed")
        self.assertEqual(0, resumed.returncode, resumed.stderr or resumed.stdout)
        payload = json.loads(queue.read_text(encoding="utf-8-sig"))
        self.assertEqual("COMPLETE", payload["status"])
        self.assertEqual(["COMPLETE", "COMPLETE"], [job["status"] for job in payload["jobs"]])
        self.assertEqual([2, 1], [job["attempts"] for job in payload["jobs"]])


if __name__ == "__main__":
    unittest.main()
