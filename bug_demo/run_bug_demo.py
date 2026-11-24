"""Injected-bug demonstration (isolated from production code).

What it does
  1. Copies the project to a TEMPORARY directory (production files are never touched).
  2. Injects a silent preprocessing bug into the copy's src/preprocessing.py.
  3. Runs the test suite on the buggy copy      -> expected: FAILS
  4. Restores the original (fixed) file in copy -> runs the suite again -> expected: PASSES
  5. Saves both full logs to reports/bug_demo/ and exits 0 only if the
     framework behaved as expected (caught the bug AND passed once fixed).

Usage:  python bug_demo/run_bug_demo.py [--bug scaling|swap|all]

Bugs (both are 'silent': nothing crashes, the model just gets wrong inputs):
  scaling : annual_income silently converted to thousands at serving time
            (a unit/scale mismatch between training and serving)
  swap    : credit_score and annual_income values swapped (mislabelled columns)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "reports" / "bug_demo"
TARGET_FILE = Path("src") / "preprocessing.py"
ANCHOR = "    return features\n"  # the single return in prepare_features()

BUGS = {
    "scaling": (
        "scale/unit bug: annual_income divided by 1000 at serving time",
        '    features["annual_income"] = features["annual_income"] / 1000  # INJECTED BUG\n' + ANCHOR,
    ),
    "swap": (
        "column mix-up bug: credit_score and annual_income values swapped",
        '    features[["credit_score", "annual_income"]] = '
        'features[["annual_income", "credit_score"]].to_numpy()  # INJECTED BUG\n' + ANCHOR,
    ),
}


def run_pytest(cwd: Path) -> subprocess.CompletedProcess:
    # --no-cov: coverage is irrelevant for the demo and would only add noise.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--no-cov", "-q", "--tb=short", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def summarize(output: str) -> str:
    """Print failing test ids, their assertion messages, and the final summary line."""
    lines = output.splitlines()
    keep = [l for l in lines if l.startswith(("FAILED", "E   ")) or " passed" in l or " failed" in l]
    return "\n".join(keep[:40])


def demo(bug: str) -> bool:
    description, buggy_code = BUGS[bug]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nINJECTED BUG '{bug}': {description}\n{'=' * 78}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "project"
        shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "reports", "htmlcov", ".coverage", "*.pyc"))
        target = work / TARGET_FILE
        original = target.read_text()
        assert original.count(ANCHOR) == 1, "Anchor line not found exactly once; update the demo patch"

        # ---- Step 1: inject the bug into the COPY only -----------------------------
        target.write_text(original.replace(ANCHOR, buggy_code))
        print(f"\n[1/2] Bug injected into temporary copy ({TARGET_FILE}). Running tests...\n")
        buggy = run_pytest(work)
        (LOG_DIR / f"{bug}_1_buggy_run.log").write_text(buggy.stdout + buggy.stderr)
        print(summarize(buggy.stdout))
        print(f"\n-> exit code {buggy.returncode} (non-zero == tests FAILED, bug detected)")

        # ---- Step 2: fix the bug (restore original) and re-run ---------------------
        target.write_text(original)
        print(f"\n[2/2] Bug fixed (original {TARGET_FILE} restored). Running tests again...\n")
        fixed = run_pytest(work)
        (LOG_DIR / f"{bug}_2_fixed_run.log").write_text(fixed.stdout + fixed.stderr)
        print(summarize(fixed.stdout))
        print(f"\n-> exit code {fixed.returncode} (0 == all tests PASS)")

    ok = buggy.returncode != 0 and fixed.returncode == 0
    print(f"\nRESULT for '{bug}': {'PASS - framework caught the bug and passed after the fix' if ok else 'UNEXPECTED BEHAVIOUR'}")
    print(f"Logs: {LOG_DIR / (bug + '_1_buggy_run.log')}\n      {LOG_DIR / (bug + '_2_fixed_run.log')}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bug", choices=[*BUGS, "all"], default="scaling")
    args = ap.parse_args()

    # Make sure a trained model exists (the bug is injected at SERVING time, so the
    # model itself is trained on correct data, exactly like the production scenario).
    if not (ROOT / "models" / "model.joblib").exists():
        subprocess.run([sys.executable, "-m", "src.model"], cwd=ROOT, check=True)

    bugs = list(BUGS) if args.bug == "all" else [args.bug]
    return 0 if all([demo(b) for b in bugs]) else 1


if __name__ == "__main__":
    sys.exit(main())
