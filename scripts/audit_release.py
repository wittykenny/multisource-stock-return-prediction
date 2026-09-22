"""Audit Git-selected release text, without displaying matched private values.

Run before git add; use --staged after staging to inspect exact index blobs.
This heuristic complements manual review; it does not certify absence of secrets.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
PATTERNS = {
    "credential-like key": r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    "private key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    "literal credential": r'''(?i)(?:api[_-]?key|secret|password|access_token|authorization|cookie)\s*[:=]\s*["'](?!your_|placeholder|example|$)[^"'\r\n]{8,}["']''',
    "bearer credential": r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{16,}",
    "database credential": r"(?i)(?:mysql|postgres(?:ql)?|mongodb)(?:\+\w+)?://[^\s/@:]+:[^\s/@]+@",
    "personal absolute path": r"[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:Users|home)/",
    "email address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "possible phone number": r"(?<![\w.])1[3-9]\d{9}(?![\w.])",
}
KEYWORDS = re.compile(r"api_key|apikey|secret|token|password|Authorization|Bearer|DashScope|OpenAI|Qwen|DeepSeek|sk-|Cookie", re.I)
ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "requirements.txt", "main.py"}
RESULT_FILES = {
    "README.md", "main_results.csv", "ablation.csv", "ablation_significance.csv",
    "reported_metrics.json", "experiment_config.json", "data_summary.json",
    "leakage_audit.json", "provenance.json",
    "optimized_55_20seed_aggregate.csv", "optimized_55_20seed_summary.csv",
    "optimized_55_20seed_summary.json", "optimized_55_20seed_config.json",
    "optimized_55_20seed_README.md",
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def allowed(path: str) -> bool:
    p = Path(path)
    if path in ROOT_FILES:
        return True
    if len(p.parts) != 2:
        return False
    folder, name = p.parts
    if folder == "stock_predictor":
        return p.suffix == ".py"
    if folder == "config":
        return p.suffix == ".yaml"
    return name in {
        "tests": {"test_strategy_significance_selection.py"},
        "scripts": {"audit_release.py"},
        "data": {"README.md"},
        "docs": {"REPRODUCIBILITY.md", "RELEASE_REPORT.md"},
        "results": RESULT_FILES,
    }.get(folder, set())


def inspect(path: str, payload: bytes) -> tuple[list[str], int]:
    issues = []
    if not allowed(path):
        issues.append(f"{path}: outside reviewed release allowlist")
    if len(payload) > MAX_BYTES:
        issues.append(f"{path}: exceeds 10 MiB limit")
        return issues, 0
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return issues + [f"{path}: non-UTF-8 payload requires manual review"], 0
    if "\0" in text:
        issues.append(f"{path}: binary content requires manual review")
    for category, expression in PATTERNS.items():
        for match in re.finditer(expression, text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{path}:{line}: {category}; inspect locally")
    return issues, len(KEYWORDS.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Scan the complete index, not working files")
    args = parser.parse_args()
    try:
        command = ("ls-files", "-z") if args.staged else ("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        paths = sorted(set(p.decode("utf-8") for p in git(*command).split(b"\0") if p))
        if not paths:
            print("FAIL: no release files selected")
            return 1
        total = 0
        findings = []
        for path in paths:
            if args.staged:
                entry = git("ls-files", "--stage", "--", path).decode("utf-8")
                if not entry.startswith(("100644 ", "100755 ")):
                    findings.append(f"{path}: non-regular index entry")
                    continue
                payload = git("show", f":{path}")
            else:
                p = ROOT / path
                if p.is_symlink() or not p.is_file():
                    findings.append(f"{path}: missing or non-regular working file")
                    continue
                payload = p.read_bytes()
            issues, count = inspect(path, payload)
            total += len(payload)
            findings.extend(issues)
            if count:
                print(f"REVIEW {path}: {count} keyword matches (may be ordinary code/documentation)")
        for issue in findings:
            print(f"FLAG {issue}")
        print(f"{'FAIL' if findings else 'PASS'}: {len(paths)} files, {total:,} bytes, {len(findings)} blocking flags")
        print("Scope: Git-selected files only; ignored local data and Git author metadata are not certified.")
        return int(bool(findings))
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        print(f"FAIL: audit could not complete ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
