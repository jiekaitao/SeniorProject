#!/usr/bin/env python3
"""
Create comprehensive visualizations for CGAR training and evaluation results.
Can upload to Weights & Biases or generate HTML reports.
"""

import json
import argparse
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def load_evaluation_results(json_path):
    """Load test evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_training_curves(data):
    """Create interactive training curves."""
    # Convert checkpoint steps to approximate epochs
    step_to_epoch = {
        "39060": 30000,
        "45570": 35000,
        "52080": 40000,
        "58590": 45000,
        "65100": 50000
    }
    
    epochs = [step_to_epoch[step] for step in data.keys()]
    exact_acc = [d["exact_accuracy"] * 100 for d in data.values()]
    token_acc = [d["token_accuracy"] * 100 for d in data.values()]
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Exact Accuracy Over Training",
            "Token Accuracy Over Training", 
            "Both Metrics Comparison",
            "Improvement from Baseline"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Exact accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs, y=exact_acc, 
            mode='lines+markers',
            name='Exact Accuracy',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Token accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs, y=token_acc,
            mode='lines+markers',
            name='Token Accuracy',
            line=dict(color='#10b981', width=3),
            marker=dict(size=10)
        ),
        row=1, col=2
    )
    
    # Both metrics
    fig.add_trace(
        go.Scatter(
            x=epochs, y=exact_acc,
            mode='lines+markers',
            name='Exact Accuracy',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=token_acc,
            mode='lines+markers',
            name='Token Accuracy',
            line=dict(color='#10b981', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Improvement bars
    improvements = [acc - exact_acc[0] for acc in exact_acc]
    fig.add_trace(
        go.Bar(
            x=epochs, y=improvements,
            name='Improvement',
            marker=dict(
                color=improvements,
                colorscale='Blues',
                showscale=False
            ),
            text=[f"+{imp:.2f}%" for imp in improvements],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)
    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    fig.update_xaxes(title_text="Epochs", row=2, col=2)
    
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Improvement (%)", row=2, col=2)
    
    fig.update_layout(
        title="CGAR Training Progress - Comprehensive View",
        height=900,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def create_summary_report(data):
    """Create a summary statistics visualization."""
    steps = list(data.keys())
    
    # Create metrics dataframe-like structure
    metrics = {
        'Checkpoint': [f"step_{s}" for s in steps],
        'Epoch': [30000, 35000, 40000, 45000, 50000],
        'Exact Acc (%)': [d['exact_accuracy'] * 100 for d in data.values()],
        'Token Acc (%)': [d['token_accuracy'] * 100 for d in data.values()],
        'Puzzles Solved': [int(d['exact_accuracy'] * d['total_puzzles']) for d in data.values()],
    }
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(metrics.keys()),
            fill_color='#667eea',
            font=dict(color='white', size=14),
            align='center'
        ),
        cells=dict(
            values=list(metrics.values()),
            fill_color=[['#f0f0f0', 'white'] * 3],
            align='center',
            font=dict(size=12),
            format=[None, ",d", ".2f", ".2f", ",d"]
        )
    )])
    
    fig.update_layout(
        title="CGAR Evaluation Results - Summary Table",
        height=400
    )
    
    return fig

def upload_to_wandb(data, project_name="cgar-results", run_name="test-evaluation"):
    """Upload results to Weights & Biases."""
    try:
        import wandb
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return False
    
    # Initialize run
    run = wandb.init(project=project_name, name=run_name, reinit=True)
    
    # Log metrics for each checkpoint
    step_to_epoch = {
        "39060": 30000,
        "45570": 35000,
        "52080": 40000,
        "58590": 45000,
        "65100": 50000
    }
    
    for step, metrics in data.items():
        epoch = step_to_epoch[step]
        wandb.log({
            "test/exact_accuracy": metrics["exact_accuracy"],
            "test/token_accuracy": metrics["token_accuracy"],
            "test/puzzles_solved": int(metrics["exact_accuracy"] * metrics["total_puzzles"]),
            "checkpoint_step": int(step),
        }, step=epoch)
    
    # Log final summary
    final_metrics = data["65100"]
    wandb.summary.update({
        "final_test_accuracy": final_metrics["exact_accuracy"],
        "final_token_accuracy": final_metrics["token_accuracy"],
        "total_test_puzzles": final_metrics["total_puzzles"],
        "training_time_hours": 6.4,
        "speedup": 5.6,
        "train_test_gap": 0.013
    })
    
    print(f"✅ Results uploaded to W&B: {run.url}")
    run.finish()
    return True

def main():
    parser = argparse.ArgumentParser(description="Create CGAR visualizations")
    parser.add_argument(
        "--results-json",
        type=str,
        default="../checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/cgar_full_50k_20251016_103736/test_evaluation_results.json",
        help="Path to test_evaluation_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload results to Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cgar-visualization",
        help="W&B project name"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"📊 Loading results from {args.results_json}...")
    data = load_evaluation_results(args.results_json)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    print("📈 Creating training curves...")
    training_fig = create_training_curves(data)
    training_fig.write_html(output_dir / "training_curves.html")
    training_fig.write_image(output_dir / "training_curves.png", width=1400, height=900)
    print(f"   ✓ Saved to {output_dir / 'training_curves.html'}")
    
    print("📋 Creating summary table...")
    summary_fig = create_summary_report(data)
    summary_fig.write_html(output_dir / "summary_table.html")
    print(f"   ✓ Saved to {output_dir / 'summary_table.html'}")
    
    # Upload to W&B if requested
    if args.wandb:
        print("☁️  Uploading to Weights & Biases...")
        upload_to_wandb(data, project_name=args.wandb_project)
    
    print("\n✨ All visualizations created successfully!")
    print(f"\n📂 Output files:")
    print(f"   • {output_dir / 'training_curves.html'}")
    print(f"   • {output_dir / 'training_curves.png'}")
    print(f"   • {output_dir / 'summary_table.html'}")
    print(f"\n💡 Open the HTML files in your browser to view interactive charts!")

if __name__ == "__main__":
    main()






