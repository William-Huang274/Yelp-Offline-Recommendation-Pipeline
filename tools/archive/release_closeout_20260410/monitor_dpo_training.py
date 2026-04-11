#!/usr/bin/env python3
"""
DPO training monitor.
Refreshes the latest log-derived training status in the terminal.
"""
import time
from datetime import datetime
from pathlib import Path


def parse_log_file(log_path: Path):
    """Parse a training log file and extract a small status summary."""
    if not log_path.exists():
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    info = {
        'config': {},
        'data': {},
        'training': {},
        'status': 'unknown',
    }

    if 'base_model=' in content:
        for line in content.split('\n'):
            if 'base_model=' in line:
                info['config']['base_model'] = line.split('base_model=')[1].split()[0]
            if 'DPO beta=' in line:
                parts = line.split('DPO beta=')[1].split(',')
                info['config']['dpo_beta'] = parts[0].strip()
            elif 'DPO ?=' in line:
                parts = line.split('DPO ?=')[1].split(',')
                info['config']['dpo_beta'] = parts[0].strip()
            if 'LR=' in line:
                parts = line.split('LR=')[1].split(',')
                info['config']['lr'] = parts[0].strip()

    if 'DPO train_pairs:' in content:
        for line in content.split('\n'):
            if 'DPO train_pairs:' in line:
                info['data']['train_pairs'] = line.split('train_pairs:')[1].split(',')[0].strip()
            if 'eval_pairs:' in line:
                info['data']['eval_pairs'] = line.split('eval_pairs:')[1].strip()

    if 'Starting DPO training' in content:
        info['status'] = 'training'
    if 'Training completed' in content or 'DONE' in content:
        info['status'] = 'completed'
    if 'ERROR' in content or 'Error' in content:
        info['status'] = 'error'

    lines = content.split('\n')
    for line in lines:
        if "'loss':" in line:
            try:
                info['training']['last_loss'] = line.split("'loss':")[1].split(',')[0].strip()
            except Exception:
                pass
        if "'epoch':" in line:
            try:
                info['training']['epoch'] = line.split("'epoch':")[1].split(',')[0].strip()
            except Exception:
                pass

    return info


def display_training_status(log_path: Path):
    """Render the current training status to the terminal."""
    print("\033[2J\033[H")
    print('=' * 70)
    print('DPO Training Monitor')
    print('=' * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Log: {log_path}')
    print()

    info = parse_log_file(log_path)

    if info is None:
        print('Waiting for training to start...')
        print('\nIf nothing happens for a while, check:')
        print('  1. Whether the training script is still running')
        print('  2. Whether the log path is correct')
        return

    if info['config']:
        print('Configuration:')
        for key, val in info['config'].items():
            print(f'  {key}: {val}')
        print()

    if info['data']:
        print('Data:')
        for key, val in info['data'].items():
            print(f'  {key}: {val}')
        print()

    status_prefix = {
        'training': '[RUN]',
        'completed': '[DONE]',
        'error': '[ERROR]',
        'unknown': '[INFO]',
    }
    print(f"{status_prefix.get(info['status'], '[INFO]')} Status: {info['status']}")

    if info['training']:
        print('\nTraining progress:')
        for key, val in info['training'].items():
            print(f'  {key}: {val}')

    print('\n' + '=' * 70)
    print('Press Ctrl+C to stop monitoring')


def monitor_training(log_path: Path, refresh_interval: int = 5):
    """Refresh the status display until interrupted."""
    try:
        while True:
            display_training_status(log_path)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print('\n\nMonitoring stopped')


def main():
    import sys

    default_log = Path('dpo_train_log.txt')
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = default_log

    print(f'Monitoring log file: {log_path}')
    print('Refresh interval: 5 seconds')
    print()

    monitor_training(log_path, refresh_interval=5)


if __name__ == '__main__':
    main()
