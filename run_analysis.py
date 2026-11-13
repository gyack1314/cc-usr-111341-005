#!/usr/bin/env python3
"""以隔离模式（python -I）运行分析 CLI 的引导脚本。

用法：
    python -I run_analysis.py analyze data/normal.json --out out
    python -I run_analysis.py simulate --reps 200
    python -I run_analysis.py check data/boundary.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from threshold_data.cli import main

if __name__ == '__main__':
    main()
