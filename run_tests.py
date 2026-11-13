#!/usr/bin/env python3
"""以隔离模式（python -I）运行全部单元测试的引导脚本。

-I 不把当前目录放入 sys.path、忽略环境变量，防止资料目录中被植入的
同名模块（如 json.py）被导入；这里显式把项目根目录加入 sys.path。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
