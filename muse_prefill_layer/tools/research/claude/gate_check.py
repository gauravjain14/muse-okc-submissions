"""Correctness gate before any okc check: the judge's 5 tests, the 8 scored shapes and nearby shapes (the final review
reruns nearby ones), two fresh graph replays each.  python research/claude/gate_check.py <candidate.py> --json out.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import check  # noqa: E402

check.EDGES.extend([
    {"batch": 4, "ctx": 70000, "layer": "sliding", "q_len": 760, "page": 1},
    {"batch": 5, "ctx": 20000, "layer": "full", "q_len": 600, "page": 1},
    {"batch": 1, "ctx": 16384, "layer": "sliding", "q_len": 15000, "page": 1},
    {"batch": 2, "ctx": 1000, "layer": "full", "q_len": 512, "page": 64},
])

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *sys.argv[1:], "--edges"]
    check.main()
