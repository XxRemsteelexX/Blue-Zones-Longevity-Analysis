#!/usr/bin/env python3
"""
Clean up packages that were installed in base Python environment
Run this script to remove packages that should only be in the virtual environment
"""

import subprocess
import sys

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def get_installed_packages():
    """Get list of installed packages in current environment"""
    stdout, stderr, returncode = run_command("pip list --format=freeze")
    if returncode == 0:
        packages = []
        for line in stdout.strip().split('\n'):
            if '==' in line:
                package_name = line.split('==')[0]
                packages.append(package_name)
        return packages
    return []

def main():
    print("🧹 Blue Zones Base Environment Cleanup")
    print("=" * 50)
    
    # Check if we're in base environment
    current_env = sys.prefix
    if 'blue_zones_env' in current_env:
        print("❌ You're in the virtual environment. Please deactivate first:")
        print("   deactivate")
        print("   python3 cleanup_base_packages.py")
        return
    
    print(f"📍 Current Python environment: {current_env}")
    
    # Packages that were likely installed for Blue Zones project
    project_packages = [
        'pandas', 'numpy', 'scipy', 'scikit-learn',
        'geopandas', 'rasterio', 'xarray', 'dask', 'pyproj', 'shapely', 'folium',
        'lightgbm', 'xgboost', 'optuna', 'shap',
        'statsmodels', 'econml', 'causalml',
        'matplotlib', 'seaborn', 'plotly', 'bokeh',
        'cdsapi', 'netcdf4', 'h5py',
        'beautifulsoup4', 'selenium',
        'jupyter', 'ipykernel', 'black', 'flake8',
        'earthengine-api', 'wbdata', 'fuzzywuzzy', 'python-levenshtein'
    ]
    
    print("\n🔍 Checking for Blue Zones packages in base environment...")
    
    installed_packages = get_installed_packages()
    packages_to_remove = []
    
    for package in project_packages:
        # Check various name formats
        package_variations = [
            package,
            package.replace('-', '_'),
            package.replace('_', '-'),
            package.lower()
        ]
        
        for variation in package_variations:
            if variation in installed_packages:
                packages_to_remove.append(variation)
                break
    
    if not packages_to_remove:
        print("✅ No Blue Zones packages found in base environment - you're clean!")
        return
    
    print(f"\n📦 Found {len(packages_to_remove)} packages to remove:")
    for pkg in packages_to_remove:
        print(f"  • {pkg}")
    
    response = input(f"\n❓ Remove these {len(packages_to_remove)} packages from base environment? (y/N): ")
    
    if response.lower() != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    print("\n🗑️ Removing packages...")
    
    for package in packages_to_remove:
        print(f"Removing {package}...", end=" ")
        stdout, stderr, returncode = run_command(f"pip uninstall -y {package}")
        
        if returncode == 0:
            print("✅")
        else:
            print(f"❌ (Error: {stderr.strip()[:50]})")
    
    print("\n✅ Base environment cleanup complete!")
    print("\n📋 Next steps:")
    print("1. Activate your virtual environment:")
    print("   cd '/home/yeblad/Desktop/Blue ZOnes'")
    print("   source blue_zones_env/bin/activate")
    print("2. All packages are properly isolated in the virtual environment")
    print("3. Run: python3 src/features/gravity_hypothesis.py")

if __name__ == "__main__":
    main()