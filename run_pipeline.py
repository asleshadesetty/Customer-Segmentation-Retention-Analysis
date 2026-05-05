"""
run_pipeline.py
---------------
Single entry point that runs the full project end to end:

  Step 1 — Generate / preprocess dataset and compute RFM
  Step 2 — K-Means segmentation + XGBoost churn prediction
  Step 3 — Generate all visualizations

Usage:
    python run_pipeline.py             # requires Kaggle dataset in data/
    python run_pipeline.py --synthetic # uses generated data
"""

import subprocess
import sys

def main():
    synthetic = "--synthetic" in sys.argv

    steps = [
        ("Generating dataset & computing RFM",
         ["src/generate_data.py"] + (["--synthetic"] if synthetic else [])),
        ("Running segmentation & churn model",
         ["src/segment_and_churn.py"]),
        ("Generating visualizations",
         ["src/visualize.py"]),
    ]

    for label, script_args in steps:
        print(f"\n{'='*60}")
        print(f"  {label}...")
        print(f"{'='*60}")
        subprocess.run([sys.executable] + script_args, check=True)

    print("\n" + "="*60)
    print("  Pipeline complete.")
    print("  Outputs  ->  outputs/")
    print("  Charts   ->  visuals/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
