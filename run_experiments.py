#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本 —— 检测环境并运行模型训练

用法:
  python run_experiments.py              # 自动检测可用模型, 运行全部
  python run_experiments.py --dry-run    # 仅打印将要运行的组合
  python run_experiments.py --model xgboost --target greenhouse_temperature
"""

import os
import sys
import json
import argparse
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_imports():
    """检测哪些库可用"""
    result = {}
    checks = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('pmdarima', 'pmdarima'),
        ('prophet', 'prophet'),
        ('xgboost', 'xgboost'),
        ('torch', 'torch'),
        ('sklearn', 'scikit-learn'),
    ]
    for name, pkg in checks:
        try:
            __import__(name)
            result[pkg] = True
        except ImportError:
            result[pkg] = False
    return result


def print_env_status(imports):
    """打印环境状态"""
    print("=" * 50)
    print("  环境状态检测")
    print("=" * 50)
    for pkg, available in imports.items():
        status = "OK" if available else "MISSING"
        print(f"  {pkg:20s}: {status}")
    print()

    if not imports['numpy'] or not imports['pandas']:
        print("[!] numpy/pandas 缺失, 请安装: pip install numpy pandas")
        sys.exit(1)


def get_available_combinations():
    """获取可用的 (模型, 目标) 组合"""
    from prepare import TARGET_CONFIG
    from train import get_available_targets, TARGET_TYPES

    imports = check_imports()

    model_map = {}
    if imports['pmdarima']:
        model_map['arima'] = ['regression']
    if imports['prophet']:
        model_map['prophet'] = ['regression']
    if imports['xgboost']:
        model_map['xgboost'] = ['regression', 'angular_regression', 'two_stage']
    if imports['torch']:
        model_map['lstm'] = ['regression', 'angular_regression', 'two_stage']
    model_map['markov'] = ['two_stage']

    targets = get_available_targets()
    if not targets:
        print("[!] 未找到预处理数据，请先运行 prepare.py")
        print(f"    期望目录: {os.path.join(CURRENT_DIR, 'autoresearch_data')}/")
        return []

    combinations = []
    for model, supported_types in model_map.items():
        for target in targets:
            target_type = TARGET_TYPES.get(target, 'regression')
            if target_type in supported_types:
                combinations.append((model, target))

    return combinations


def main():
    parser = argparse.ArgumentParser(description='快速开始脚本')
    parser.add_argument('--dry-run', action='store_true', help='仅打印运行计划')
    parser.add_argument('--model', type=str, help='指定模型')
    parser.add_argument('--target', type=str, help='指定目标')
    parser.add_argument('--skip-prepare', action='store_true', help='跳过 prepare.py')
    parser.add_argument('--results', type=str, default='results.csv', help='结果文件')
    args = parser.parse_args()

    imports = check_imports()
    print_env_status(imports)

    if not args.skip_prepare:
        print("[1] 运行 prepare.py 准备数据...")
        result = subprocess.run(
            [sys.executable, os.path.join(CURRENT_DIR, 'prepare.py')],
            capture_output=True, text=True, cwd=CURRENT_DIR
        )
        if result.returncode != 0:
            print("[!] prepare.py 运行失败:")
            print(result.stderr[-500:])
        else:
            print("    prepare.py 完成")

    if args.model and args.target:
        combinations = [(args.model, args.target)]
    else:
        combinations = get_available_combinations()

    if not combinations:
        print("\n[!] 无可用 (模型×目标) 组合")
        print("    请确保已运行 prepare.py 生成数据")
        print("    并安装所需依赖: pip install numpy pandas pmdarima prophet xgboost torch")
        return

    print(f"\n[2] 计划运行 {len(combinations)} 个实验:")
    print("-" * 50)
    for model, target in combinations:
        print(f"    {model:10s} -> {target}")
    print("-" * 50)

    if args.dry_run:
        print("\n[DRY-RUN] 不实际运行训练")
        return

    print(f"\n[3] 开始训练...")
    success = 0
    fail = 0

    for model, target in combinations:
        cmd = [
            sys.executable, os.path.join(CURRENT_DIR, 'train.py'),
            '--model', model,
            '--target', target,
            '--results', args.results,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=CURRENT_DIR)

        if result.returncode == 0:
            success += 1
            last_line = [l for l in result.stdout.strip().split('\n') if l.strip()][-1] if result.stdout.strip() else ''
            print(f"  [OK] {model:10s} -> {target:30s} {last_line[:80]}")
        else:
            fail += 1
            print(f"  [FAIL] {model:10s} -> {target:30s}")
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-3:]:
                if line.strip():
                    print(f"         {line.strip()[:100]}")

    print(f"\n[4] 训练完成: {success} 成功, {fail} 失败")
    if os.path.exists(os.path.join(CURRENT_DIR, args.results)):
        print(f"    结果文件: {os.path.join(CURRENT_DIR, args.results)}")


if __name__ == '__main__':
    main()