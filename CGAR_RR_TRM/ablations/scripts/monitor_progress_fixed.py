#!/usr/bin/env python3
"""
CGAR Ablation Study - Progress Monitor (Fixed)
Monitors ablation experiments in real-time
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

def parse_log_progress(log_file):
    """Extract current batch/epoch from log file"""
    try:
        # Read last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()[-50:]
        
        # Look for tqdm progress pattern: "X/Y [time<remaining, speed]"
        for line in reversed(lines):
            # Match pattern like "  2089/39062 [time<time, 3.47it/s]"
            match = re.search(r'\s+(\d+)/(\d+)\s+\[', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                progress = (current / total) * 100
                
                # Extract speed if present
                speed_match = re.search(r'([\d.]+)(it/s|s/it)', line)
                if speed_match:
                    speed_val = float(speed_match.group(1))
                    speed_unit = speed_match.group(2)
                    if speed_unit == 's/it':
                        speed = f"{speed_val:.2f}s/it"
                    else:
                        speed = f"{speed_val:.2f}it/s"
                else:
                    speed = "N/A"
                
                return {
                    'current': current,
                    'total': total,
                    'progress': progress,
                    'speed': speed,
                    'status': 'Running'
                }
        
        # Check for error patterns
        for line in lines:
            if 'Error' in line or 'error' in line:
                return {
                    'current': 0,
                    'total': 39062,
                    'progress': 0,
                    'speed': 'N/A',
                    'status': 'Error',
                    'error': line.strip()[:100]
                }
        
        return {
            'current': 0,
            'total': 39062,
            'progress': 0,
            'speed': 'N/A',
            'status': 'Starting'
        }
    
    except Exception as e:
        return {
            'current': 0,
            'total': 39062,
            'progress': 0,
            'speed': 'N/A',
            'status': f'Error: {str(e)[:50]}'
        }

def find_latest_log(experiment_name):
    """Find the latest log file for an experiment"""
    log_dir = Path("/data/TRM/ablations/logs")
    pattern = f"{experiment_name}_*.log"
    
    logs = list(log_dir.glob(pattern))
    if not logs:
        return None
    
    # Return most recent
    return max(logs, key=lambda p: p.stat().st_mtime)

def check_tmux_session(session_name):
    """Check if tmux session is running"""
    import subprocess
    try:
        result = subprocess.run(
            ['tmux', 'ls'],
            capture_output=True,
            text=True
        )
        return session_name in result.stdout
    except:
        return False

def main():
    experiments = [
        ('baseline', 'abl_baseline'),
        ('curriculum', 'abl_curriculum'),
        ('hierarchical', 'abl_hierarchical'),
        ('cgar_full', 'abl_cgar'),
    ]
    
    print("\n" + "="*80)
    print(" CGAR Ablation Study - Live Progress Monitor")
    print("="*80 + "\n")
    
    print(f"{'Experiment':<20} {'Status':<12} {'Progress':<12} {'Batch':<15} {'Speed':<15}")
    print("-" * 80)
    
    for exp_name, session_name in experiments:
        # Check if session is running
        session_active = check_tmux_session(session_name)
        
        # Find log file
        log_file = find_latest_log(exp_name)
        
        if not log_file:
            status = "No log"
            progress_str = "0.0%"
            batch_str = "0/39062"
            speed_str = "N/A"
        else:
            # Parse progress
            info = parse_log_progress(log_file)
            status = "🟢 Running" if session_active else "⚠️  Stopped"
            if info['status'] == 'Error':
                status = "❌ Error"
            progress_str = f"{info['progress']:.1f}%"
            batch_str = f"{info['current']}/{info['total']}"
            speed_str = info['speed']
        
        print(f"{exp_name:<20} {status:<12} {progress_str:<12} {batch_str:<15} {speed_str:<15}")
    
    print("-" * 80)
    print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo view a specific experiment:")
    print("  tmux attach -t abl_baseline    # View baseline")
    print("  tmux attach -t abl_cgar        # View CGAR")
    print("\nPress Ctrl+B then D to detach from tmux\n")

if __name__ == "__main__":
    main()

