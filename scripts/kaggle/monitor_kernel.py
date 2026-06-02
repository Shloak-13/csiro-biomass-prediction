from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime


TERMINAL_STATES = {"complete", "error", "cancelled", "failed"}


def kernel_status(kernel: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output)
    print(output)
    lowered = output.lower()
    for state in TERMINAL_STATES | {"running", "queued"}:
        if state in lowered:
            return state
    return "unknown"


def monitor(kernel: str, interval: int, timeout_minutes: int) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] polling {kernel}")
        status = kernel_status(kernel)
        if status in TERMINAL_STATES:
            return status
        if time.time() > deadline:
            raise TimeoutError(f"Timed out after {timeout_minutes} minutes waiting for {kernel}")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Kaggle kernel execution status.")
    parser.add_argument("--kernel", required=True, help="<owner>/<kernel-slug>")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--timeout-minutes", type=int, default=720)
    args = parser.parse_args()
    status = monitor(args.kernel, args.interval, args.timeout_minutes)
    print(f"final_status={status}")
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
