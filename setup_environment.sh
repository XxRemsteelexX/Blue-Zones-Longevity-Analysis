#!/bin/bash
# Blue Zones Environment Setup Script

echo "🌍 Setting up Blue Zones Quantified Environment..."

# Navigate to project directory
cd "/home/yeblad/Desktop/Blue ZOnes"

# Check if virtual environment exists, if not create it
if [ ! -d "blue_zones_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv blue_zones_env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source blue_zones_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing project dependencies..."
pip install -r requirements.txt

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed,results}/{life_expectancy,climate,population,socioeconomic,amenities,elevation}
mkdir -p outputs/{figures,tables,reports}
mkdir -p logs

# Set executable permissions for scripts
echo "🔧 Setting script permissions..."
find scripts/ -name "*.py" -exec chmod +x {} \;

# Test installation
echo "🧪 Testing core imports..."
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import pandas as pd
    print(f'✅ pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ pandas: {e}')

try:
    import numpy as np
    print(f'✅ numpy: {np.__version__}')
except ImportError as e:
    print(f'❌ numpy: {e}')

try:
    import geopandas as gpd
    print(f'✅ geopandas: {gpd.__version__}')
except ImportError as e:
    print(f'❌ geopandas: {e}')

try:
    import sklearn
    print(f'✅ scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'❌ scikit-learn: {e}')

try:
    import plotly
    print(f'✅ plotly: {plotly.__version__}')
except ImportError as e:
    print(f'❌ plotly: {e}')

print('\n🔬 Testing Blue Zones modules...')
sys.path.append('src')

try:
    from features.gravity_hypothesis import GravityLongevityAnalyzer
    print('✅ Gravity hypothesis module')
except ImportError as e:
    print(f'❌ Gravity hypothesis: {e}')

try:
    from models.panel_fe import PanelFixedEffects
    print('✅ Panel fixed effects module')
except ImportError as e:
    print(f'❌ Panel FE: {e}')

try:
    from models.spatial_spillovers import SpatialSpilloverAnalyzer
    print('✅ Spatial spillovers module')
except ImportError as e:
    print(f'❌ Spatial spillovers: {e}')
"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate environment in future sessions:"
echo "  cd '/home/yeblad/Desktop/Blue ZOnes'"
echo "  source blue_zones_env/bin/activate"
echo ""
echo "To run analysis:"
echo "  python3 src/features/gravity_hypothesis.py"
echo "  python3 scripts/download_sample_data.py"
echo ""
echo "To deactivate:"
echo "  deactivate"