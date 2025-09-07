#!/usr/bin/env python3
"""
Run the complete Blue Zones Quantified analysis pipeline
"""
import sys
import logging
import subprocess
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, load_config


def main():
    """Run the complete analysis pipeline"""
    
    # Setup
    logger = setup_logging("INFO")
    config = load_config("config/config.yaml")
    
    logger.info("🌍 Starting Blue Zones Quantified Full Analysis Pipeline")
    logger.info("=" * 70)
    
    pipeline_start_time = time.time()
    
    # Define pipeline steps
    pipeline_steps = [
        {
            'name': 'Build Global Grid System',
            'script': '01_build_grid.py',
            'description': 'Create 5km global grid with Blue Zone labels'
        },
        {
            'name': 'ETL Climate Data',
            'script': '02_etl_climate.py', 
            'description': 'Extract and process climate features from ERA5'
        },
        {
            'name': 'ETL Demographics Data',
            'script': '03_etl_demographics.py',
            'description': 'Process population, socioeconomic, and nightlights data'
        },
        {
            'name': 'Engineer Features',
            'script': '04_engineer_features.py',
            'description': 'Generate comprehensive feature set including lifestyle proxies'
        },
        {
            'name': 'Add Life Expectancy',
            'script': '05_add_life_expectancy.py',
            'description': 'Integrate life expectancy outcome data from IHME GBD'
        },
        {
            'name': 'Matched Comparison Analysis',
            'script': '06_matched_comparison.py',
            'description': 'Estimate Blue Zone treatment effects using causal inference'
        },
        {
            'name': 'Blue Zone Classifier',
            'script': '07_blue_zone_classifier.py',
            'description': 'Train ML classifier and identify candidate regions'
        },
        {
            'name': 'Forecasting Models',
            'script': '08_forecasting_models.py',
            'description': 'Build life expectancy forecasting models with uncertainty'
        },
        {
            'name': 'Create Visualizations',
            'script': '09_create_visualizations.py',
            'description': 'Generate interactive dashboard and analysis plots'
        }
    ]
    
    # Run pipeline steps
    results = []
    
    for i, step in enumerate(pipeline_steps, 1):
        logger.info(f"\n📊 Step {i}/{len(pipeline_steps)}: {step['name']}")
        logger.info(f"Description: {step['description']}")
        logger.info("-" * 50)
        
        step_start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, f"scripts/{step['script']}"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            step_duration = time.time() - step_start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {step['name']} completed successfully ({step_duration:.1f}s)")
                results.append({
                    'step': step['name'],
                    'status': 'success',
                    'duration': step_duration
                })
            else:
                logger.error(f"❌ {step['name']} failed ({step_duration:.1f}s)")
                logger.error(f"Error output: {result.stderr}")
                results.append({
                    'step': step['name'],
                    'status': 'failed',
                    'duration': step_duration,
                    'error': result.stderr
                })
                
                # Ask user whether to continue
                if not should_continue_after_error(step['name'], logger):
                    break
                    
        except Exception as e:
            step_duration = time.time() - step_start_time
            logger.error(f"❌ {step['name']} failed with exception ({step_duration:.1f}s): {e}")
            results.append({
                'step': step['name'],
                'status': 'failed',
                'duration': step_duration,
                'error': str(e)
            })
            
            if not should_continue_after_error(step['name'], logger):
                break
    
    # Pipeline summary
    total_duration = time.time() - pipeline_start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("🏁 PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 70)
    
    successful_steps = [r for r in results if r['status'] == 'success']
    failed_steps = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info(f"Successful steps: {len(successful_steps)}/{len(results)}")
    logger.info(f"Failed steps: {len(failed_steps)}/{len(results)}")
    
    if successful_steps:
        logger.info("\n✅ Successful Steps:")
        for result in successful_steps:
            logger.info(f"  • {result['step']} ({result['duration']:.1f}s)")
    
    if failed_steps:
        logger.info("\n❌ Failed Steps:")
        for result in failed_steps:
            logger.info(f"  • {result['step']} ({result['duration']:.1f}s)")
    
    # Output locations
    logger.info("\n📁 Output Locations:")
    logger.info("  • Processed data: data/processed/")
    logger.info("  • Features: data/features/")
    logger.info("  • Models: data/models/")
    logger.info("  • Results: data/outputs/")
    logger.info("  • Visualizations: data/outputs/visualizations/")
    
    # Key deliverables
    logger.info("\n🎯 Key Deliverables:")
    key_outputs = [
        "data/outputs/visualizations/blue_zones_dashboard.html",
        "data/outputs/blue_zone_predictions.csv",
        "data/outputs/life_expectancy_forecasts.csv",
        "data/outputs/matched_comparison_summary.csv",
        "data/outputs/candidate_blue_zones.csv"
    ]
    
    for output in key_outputs:
        output_path = Path(output)
        if output_path.exists():
            logger.info(f"  ✅ {output}")
        else:
            logger.info(f"  ❌ {output} (not generated)")
    
    # Final recommendations
    logger.info("\n💡 Next Steps:")
    if len(successful_steps) == len(pipeline_steps):
        logger.info("  • Review the interactive dashboard in data/outputs/visualizations/")
        logger.info("  • Analyze candidate Blue Zone regions")
        logger.info("  • Validate findings with domain experts")
        logger.info("  • Consider additional feature engineering based on results")
    else:
        logger.info("  • Address failed pipeline steps")
        logger.info("  • Check data availability and format requirements")
        logger.info("  • Review error logs for debugging information")
    
    # Return appropriate exit code
    if len(failed_steps) == 0:
        logger.info("\n🎉 Pipeline completed successfully!")
        return 0
    elif len(successful_steps) > len(failed_steps):
        logger.info("\n⚠️  Pipeline completed with some failures")
        return 1
    else:
        logger.info("\n💥 Pipeline failed")
        return 2


def should_continue_after_error(step_name: str, logger: logging.Logger) -> bool:
    """Ask user whether to continue pipeline after error"""
    
    # For automated runs, continue by default
    import os
    if os.environ.get('BLUE_ZONES_AUTO_CONTINUE', '').lower() == 'true':
        logger.info("Auto-continuing due to BLUE_ZONES_AUTO_CONTINUE environment variable")
        return True
    
    # In interactive mode, continue for non-critical steps
    non_critical_steps = [
        'ETL Climate Data',
        'ETL Demographics Data', 
        'Create Visualizations'
    ]
    
    if step_name in non_critical_steps:
        logger.warning(f"Step '{step_name}' failed but is non-critical, continuing...")
        return True
    
    logger.warning(f"Critical step '{step_name}' failed")
    return False


if __name__ == "__main__":
    sys.exit(main())