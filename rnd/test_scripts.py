#!/usr/bin/env python3
"""
Test script to verify Python scripts can be imported and basic functionality works
"""

import sys
import os

def test_imports():
    """Test if we can import the main modules"""
    print("Testing imports...")
    
    try:
        # Test basic Python imports
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    return True

def test_script_syntax():
    """Test if our scripts have valid syntax"""
    print("\nTesting script syntax...")
    
    scripts = [
        'regional_models_analysis.py',
        'segment_models_analysis.py', 
        'forecasting_and_outputs.py',
        'run_analysis.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            try:
                with open(script, 'r') as f:
                    code = f.read()
                compile(code, script, 'exec')
                print(f"✅ {script} - syntax OK")
            except SyntaxError as e:
                print(f"❌ {script} - syntax error: {e}")
                return False
        else:
            print(f"⚠️ {script} - file not found")
    
    return True

def test_data_files():
    """Test if required data files exist"""
    print("\nTesting data files...")
    
    required_files = [
        'data/monthly_sales_summary.csv',
        'outputs/processed_weather_data/AP_weather_timeseries.csv',
        'outputs/processed_weather_data/KA_weather_timeseries.csv',
        'outputs/processed_weather_data/KL_weather_timeseries.csv',
        'outputs/processed_weather_data/TL_weather_timeseries.csv',
        'outputs/processed_weather_data/TN_weather_timeseries.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        return False
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING AC SALES & WEATHER ANALYSIS SCRIPTS")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test syntax
    syntax_ok = test_script_syntax()
    
    # Test data files
    data_ok = test_data_files()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"Data files: {'✅ PASS' if data_ok else '❌ FAIL'}")
    
    if imports_ok and syntax_ok and data_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("Scripts are ready to run (after installing dependencies)")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please fix issues before running the analysis")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)