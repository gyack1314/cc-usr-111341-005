"""命令行入口：资料检查、阈值分析与模拟验证。

不带子命令时保留历史行为（等同于 check），旧调用方式不受影响：
    python -m threshold_data.cli data/normal.json
新分析入口：
    python -m threshold_data.cli analyze data/normal.json --out out
    python -m threshold_data.cli simulate --reps 200
"""
import argparse
import json
from .io import load, DataError
from .domain import inspect_tables
from .pipeline import analyze
from . import simulation


def _print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2,
                     default=lambda v: 'NaN' if isinstance(v, float)
                     and v != v else str(v)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='资格阈值附近的扶持效果识别')
    sub = parser.add_subparsers(dest='command')

    check = sub.add_parser('check', help='资料身份与完整性检查（默认）')
    check.add_argument('manifest')

    analysis = sub.add_parser('analyze', help='运行完整诊断与效果识别')
    analysis.add_argument('manifest')
    analysis.add_argument('--out', default='out')
    analysis.add_argument('--horizon', type=int, default=90)
    analysis.add_argument('--half-width', type=float, default=2.0)

    sim = sub.add_parser('simulate', help='独立 DGP 偏差与覆盖率验证')
    sim.add_argument('--reps', type=int, default=200)
    sim.add_argument('--out', default=None)

    raw_args = list(argv) if argv is not None else None
    import sys
    positional = raw_args if raw_args is not None else sys.argv[1:]
    # 兼容旧位置参数：python -m threshold_data.cli <manifest>
    # 子解析器会拒绝裸位置参数，这里在 argparse 之前手动改写为 check。
    if positional and positional[0] not in ('check', 'analyze', 'simulate',
                                            '-h', '--help'):
        positional = ['check'] + positional
    legacy = parser.parse_args(positional)
    if legacy.command is None:
        parser.print_help()
        return
    try:
        if legacy.command == 'check':
            _print_json(inspect_tables(load(legacy.manifest)))
        elif legacy.command == 'analyze':
            findings, target = analyze(
                legacy.manifest, out_dir=legacy.out,
                horizon_days=legacy.horizon,
                default_half_width=legacy.half_width)
            print(f'分析完成，产物目录：{target}')
            _print_json({k: findings[k] for k in
                         ('package_path', 'sample', 'support',
                          'frozen_score_audit', 'scope_and_limits')})
        elif legacy.command == 'simulate':
            result = simulation.run_all(reps_continuous=legacy.reps,
                                        reps_weak=legacy.reps,
                                        reps_manip=max(50, legacy.reps // 2),
                                        reps_discrete=max(50, legacy.reps // 2))
            _print_json(result)
            if legacy.out:
                with open(legacy.out, 'w', encoding='utf-8') as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
    except (DataError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
