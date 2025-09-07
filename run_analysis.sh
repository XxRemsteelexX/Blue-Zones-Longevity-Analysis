#!/bin/bash
# Blue Zones Analysis Runner
# Complete workflow for Blue Zones research

echo "🌍 Blue Zones Quantified: Complete Analysis Workflow"
echo "=" * 60

# Navigate to project directory
cd "/home/yeblad/Desktop/Blue ZOnes"

# Activate virtual environment
echo "🔄 Activating environment..."
source blue_zones_env/bin/activate

# Check if data exists, if not download sample data
if [ ! -f "data/raw/socioeconomic/world_bank_sample.csv" ]; then
    echo "📥 Downloading sample data..."
    python3 scripts/download_sample_data.py
fi

echo ""
echo "🎯 ANALYSIS OPTIONS:"
echo "1. Interactive Jupyter Analysis (Recommended for exploration)"
echo "2. Run Complete Analysis Pipeline"
echo "3. Test Gravity Hypothesis Only"
echo "4. Generate Research Report"
echo ""

read -p "Choose option (1-4): " option

case $option in
    1)
        echo "🚀 Starting Jupyter Lab..."
        echo "📝 Recommended notebooks to start with:"
        echo "   • notebooks/01_Data_Acquisition_Demo.ipynb"
        echo "   • notebooks/02_Geospatial_Data_Processing.ipynb"
        echo "   • notebooks/03_Gravity_Analysis.ipynb (create this!)"
        echo ""
        echo "💡 Access at: http://localhost:8888"
        jupyter lab --no-browser --ip=0.0.0.0 --port=8888
        ;;
    2)
        echo "⚙️ Running complete analysis pipeline..."
        python3 scripts/run_complete_analysis.py
        ;;
    3)
        echo "🌍 Testing gravity hypothesis..."
        python3 src/features/gravity_hypothesis.py
        ;;
    4)
        echo "📊 Generating research report..."
        python3 scripts/generate_research_report.py
        ;;
    *)
        echo "❌ Invalid option. Exiting."
        ;;
esac