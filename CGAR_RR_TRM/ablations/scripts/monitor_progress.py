#!/usr/bin/env python3
"""
Monitor progress of all ablation experiments
Shows real-time status of each experiment
"""
import os
import sys
import time
import json
import glob
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

ABLATION_ROOT = Path("/data/TRM/ablations")
RESULTS_DIR = ABLATION_ROOT / "results"
LOGS_DIR = ABLATION_ROOT / "logs"

EXPERIMENTS = {
    "Component Ablations": [
        ("baseline", "Baseline (neither)"),
        ("curriculum", "Curriculum only"),
        ("hierarchical", "Hierarchical only"),
        ("cgar_full", "Full CGAR"),
    ],
    "Decay Sweep": [
        (f"decay_{d}", f"Decay={d}") for d in ["0.5", "0.6", "0.7", "0.8", "0.9"]
    ]
}

def get_latest_checkpoint(exp_dir):
    """Find latest checkpoint in experiment directory"""
    checkpoints = glob.glob(str(exp_dir / "step_*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def extract_accuracy_from_log(log_file):
    """Extract latest accuracy from log file"""
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Look for accuracy lines (implementation depends on log format)
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if 'exact_accuracy' in line or 'accuracy' in line:
                    # Try to extract accuracy value
                    try:
                        if 'exact_accuracy' in line:
                            parts = line.split('exact_accuracy')
                            if len(parts) > 1:
                                acc_str = parts[1].split()[0].strip(':,')
                                return float(acc_str) * 100
                    except:
                        continue
    except:
        pass
    return None

def get_experiment_status(exp_name):
    """Get status of an experiment"""
    exp_dir = RESULTS_DIR / exp_name
    
    status = {
        "running": False,
        "progress": 0,
        "accuracy": None,
        "eta": "Unknown",
        "checkpoint": None
    }
    
    # Check if results directory exists
    if not exp_dir.exists():
        return status
    
    # Check for checkpoints
    checkpoint = get_latest_checkpoint(exp_dir)
    if checkpoint:
        status["checkpoint"] = Path(checkpoint).name
        status["running"] = True
        
        # Estimate progress from checkpoint number
        # Assuming step_XXXXX format and 30K total steps
        try:
            step = int(Path(checkpoint).name.split('_')[1])
            status["progress"] = min(100, int((step / 30000) * 100))
        except:
            pass
    
    # Check logs for accuracy
    log_files = glob.glob(str(LOGS_DIR / f"{exp_name}_*.log"))
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        acc = extract_accuracy_from_log(Path(latest_log))
        if acc:
            status["accuracy"] = acc
    
    return status

def create_status_table():
    """Create rich table with experiment statuses"""
    table = Table(title="CGAR Ablation Study Progress", show_header=True, header_style="bold magenta")
    
    table.add_column("Experiment", style="cyan", width=30)
    table.add_column("Status", style="green", width=10)
    table.add_column("Progress", justify="right", width=10)
    table.add_column("Accuracy", justify="right", width=12)
    table.add_column("Checkpoint", width=15)
    
    for category, experiments in EXPERIMENTS.items():
        table.add_row(f"[bold yellow]{category}[/]", "", "", "", "")
        
        for exp_name, exp_desc in experiments:
            status = get_experiment_status(exp_name)
            
            if status["running"]:
                status_icon = "🟢 Running"
                progress_str = f"{status['progress']}%"
            elif status["checkpoint"]:
                status_icon = "✅ Done"
                progress_str = "100%"
            else:
                status_icon = "⏳ Waiting"
                progress_str = "0%"
            
            acc_str = f"{status['accuracy']:.2f}%" if status['accuracy'] else "N/A"
            ckpt_str = status['checkpoint'] if status['checkpoint'] else "None"
            
            table.add_row(
                f"  {exp_desc}",
                status_icon,
                progress_str,
                acc_str,
                ckpt_str
            )
    
    return table

def main():
    """Main monitoring loop"""
    try:
        with Live(create_status_table(), refresh_per_second=0.5, console=console) as live:
            while True:
                live.update(create_status_table())
                time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("Installing rich for better display...")
        os.system("cd /data/TRM && source TRMvenv/bin/activate && uv pip install rich")
        print("Please run the script again")
        sys.exit(1)
    
    console.print("[bold green]CGAR Ablation Study Monitor[/bold green]")
    console.print("Press Ctrl+C to stop\n")
    
    main()

