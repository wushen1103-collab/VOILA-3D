from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pid-file", action="append", default=[])
    p.add_argument("--out-csv", required=True)
    p.add_argument("--interval-sec", type=float, default=10.0)
    p.add_argument("--max-seconds", type=float, default=0.0)
    return p.parse_args()


def _read_pid(path: str | Path) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def _ps_row(pid: int) -> dict[str, str | int | float] | None:
    out = _run(["ps", "-p", str(pid), "-o", "pid=,pcpu=,rss=,vsz=,etime=,cmd="]).strip()
    if not out:
        return None
    parts = out.split(None, 5)
    if len(parts) < 6:
        return None
    return {
        "kind": "process",
        "pid": int(parts[0]),
        "gpu_index": "",
        "gpu_uuid": "",
        "gpu_memory_mb": "",
        "gpu_util_pct": "",
        "pcpu": float(parts[1]),
        "rss_kb": int(parts[2]),
        "vsz_kb": int(parts[3]),
        "etime": parts[4],
        "cmd": parts[5],
    }


def _gpu_rows() -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    gpu_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in gpu_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "kind": "gpu",
                "pid": "",
                "gpu_index": parts[0],
                "gpu_uuid": parts[1],
                "gpu_memory_mb": parts[2],
                "gpu_total_memory_mb": parts[3],
                "gpu_util_pct": parts[4],
                "pcpu": "",
                "rss_kb": "",
                "vsz_kb": "",
                "etime": "",
                "cmd": "",
            }
        )
    app_out = _run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    for line in app_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append(
            {
                "kind": "gpu_process",
                "pid": parts[0],
                "gpu_index": "",
                "gpu_uuid": parts[1],
                "gpu_memory_mb": parts[2],
                "gpu_total_memory_mb": "",
                "gpu_util_pct": "",
                "pcpu": "",
                "rss_kb": "",
                "vsz_kb": "",
                "etime": "",
                "cmd": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "kind",
        "pid",
        "gpu_index",
        "gpu_uuid",
        "gpu_memory_mb",
        "gpu_total_memory_mb",
        "gpu_util_pct",
        "pcpu",
        "rss_kb",
        "vsz_kb",
        "etime",
        "cmd",
    ]
    start = time.time()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while True:
            now = datetime.now(timezone.utc).isoformat()
            pids = [pid for pid in (_read_pid(path) for path in args.pid_file) if pid is not None]
            alive_pids = [pid for pid in pids if _pid_alive(pid)]
            for pid in alive_pids:
                row = _ps_row(pid)
                if row is not None:
                    row["timestamp"] = now
                    writer.writerow(row)
            for row in _gpu_rows():
                row["timestamp"] = now
                writer.writerow(row)
            f.flush()
            if not alive_pids:
                break
            if args.max_seconds and time.time() - start >= args.max_seconds:
                break
            time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    main()
