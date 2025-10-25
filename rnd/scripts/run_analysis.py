#!/usr/bin/env python3
"""
Main Runner Script for AC Sales & Weather Analysis

This script executes all phases of the analysis sequentially:
1. Phase 1-4: Combined scope models (weather_sales_analysis.ipynb)
2. Phase 5: Regional models analysis
3. Phase 6: Segment models analysis  
4. Phase 7: Forecasting and outputs

Usage:
    python run_analysis.py [--phases 1,2,3,4,5,6,7] [--skip-notebook]
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_python_script(script_path, phase_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING {phase_name}")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now()}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"\n✅ {phase_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {phase_name}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {phase_name}: {e}")
        return False

def run_notebook(notebook_path, phase_name):
    """Run a Jupyter notebook using nbconvert"""
    print(f"\n{'='*80}")
    print(f"RUNNING {phase_name}")
    print(f"{'='*80}")
    print(f"Notebook: {notebook_path}")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Convert notebook to script and run
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'script', '--execute', 
            '--ExecutePreprocessor.timeout=3600', notebook_path
        ], capture_output=True, text=True, check=True)
        
        print(f"\n✅ {phase_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error in {phase_name}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {phase_name}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'statsmodels', 
        'prophet', 'tensorflow', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    # Skip pmdarima check due to numpy compatibility issues
    print(f"  ⚠️ pmdarima (skipped - known compatibility issues)")
    
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("Core dependencies are available!")
    print("Note: pmdarima has compatibility issues but scripts will use alternative implementations.")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run AC Sales & Weather Analysis')
    parser.add_argument('--phases', default='1,2,3,4,5,6,7', 
                       help='Comma-separated list of phases to run (default: 1,2,3,4,5,6,7)')
    parser.add_argument('--skip-notebook', action='store_true',
                       help='Skip notebook execution (Phases 1-4)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Only check dependencies and exit')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AC SALES & WEATHER ANALYSIS - MAIN RUNNER")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Phases to run: {args.phases}")
    print(f"Skip notebook: {args.skip_notebook}")
    
    # Check dependencies
    if not check_dependencies():
        if args.check_deps:
            return 1
        print("\nDependencies check failed. Please install missing packages.")
        return 1
    
    if args.check_deps:
        print("\nDependencies check passed!")
        return 0
    
    # Parse phases
    try:
        phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
    except ValueError:
        print("Error: Invalid phase format. Use comma-separated integers (e.g., 1,2,3,4,5,6,7)")
        return 1
    
    # Ensure output directories exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/charts', exist_ok=True)
    os.makedirs('outputs/forecasts', exist_ok=True)
    
    # Track execution results
    results = {}
    
    # Phase 1-4: Combined scope models (notebook)
    if any(phase in phases_to_run for phase in [1, 2, 3, 4]) and not args.skip_notebook:
        notebook_path = 'weather_sales_analysis.ipynb'
        if os.path.exists(notebook_path):
            success = run_notebook(notebook_path, "PHASES 1-4: COMBINED SCOPE MODELS")
            results['phases_1_4'] = success
        else:
            print(f"⚠️ Notebook not found: {notebook_path}")
            results['phases_1_4'] = False
    
    # Phase 5: Regional models analysis
    if 5 in phases_to_run:
        script_path = 'regional_models_analysis.py'
        if os.path.exists(script_path):
            success = run_python_script(script_path, "PHASE 5: REGIONAL MODELS ANALYSIS")
            results['phase_5'] = success
        else:
            print(f"⚠️ Script not found: {script_path}")
            results['phase_5'] = False
    
    # Phase 6: Segment models analysis
    if 6 in phases_to_run:
        script_path = 'segment_models_analysis.py'
        if os.path.exists(script_path):
            success = run_python_script(script_path, "PHASE 6: SEGMENT MODELS ANALYSIS")
            results['phase_6'] = success
        else:
            print(f"⚠️ Script not found: {script_path}")
            results['phase_6'] = False
    
    # Phase 7: Forecasting and outputs
    if 7 in phases_to_run:
        script_path = 'forecasting_and_outputs.py'
        if os.path.exists(script_path):
            success = run_python_script(script_path, "PHASE 7: FORECASTING AND OUTPUTS")
            results['phase_7'] = success
        else:
            print(f"⚠️ Script not found: {script_path}")
            results['phase_7'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    
    total_phases = len(results)
    successful_phases = sum(1 for success in results.values() if success)
    
    for phase, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{phase.upper()}: {status}")
    
    print(f"\nOverall: {successful_phases}/{total_phases} phases completed successfully")
    
    if successful_phases == total_phases:
        print("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        print("- outputs/forecasts/ - Forecast CSV files")
        print("- outputs/charts/ - Visualization files")
        print("- outputs/models/ - Trained model files")
        print("- outputs/results_summary.md - Comprehensive summary")
        return 0
    else:
        print(f"\n⚠️ {total_phases - successful_phases} phase(s) failed. Check logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
