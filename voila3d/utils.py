from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def available_cpu_threads(reserve: int = 30, cap: int | None = None) -> int:
    count = os.cpu_count() or 1
    usable = max(1, count - reserve)
    if cap is not None:
        usable = min(usable, cap)
    return usable


def write_markdown_table(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path = Path(path)
    header = list(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
