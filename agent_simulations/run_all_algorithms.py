#!/usr/bin/env python3
"""
Wrapper for Optuna tuning orchestration.

USAGE:
  bash:   bash run_tuning.sh              (RECOMMENDED - efficient logging)
  python: python run_all_algorithms.py    (legacy wrapper - calls bash script)
  python: python generate_comparison_plots.py  (generate plots from existing results)

The bash script is HIGHLY RECOMMENDED for:
  ✓ Comprehensive logging per algorithm
  ✓ Timing information
  ✓ Error handling and recovery
  ✓ Progress tracking
  ✓ Automatic result parsing and validation
"""
import subprocess
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    bash_script = script_dir / 'run_tuning.sh'
    
    if not bash_script.exists():
        print(f"ERROR: Bash script not found: {bash_script}")
        print("\nUsage:")
        print("  bash run_tuning.sh                    (RECOMMENDED)")
        print("  python run_all_algorithms.py          (this wrapper)")
        return 1
    
    print("\n" + "="*80)
    print("OPTUNA TUNING ORCHESTRATION - BASH WRAPPER")
    print("="*80)
    print("\nExecuting bash script for efficient logging and orchestration...\n")
    
    # Run bash script
    try:
        result = subprocess.run(
            ['bash', str(bash_script)],
            cwd=script_dir,
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"ERROR: Failed to execute bash script: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
