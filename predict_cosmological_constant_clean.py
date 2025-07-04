#!/usr/bin/env python3
"""
Main Prediction Script - LQG Cosmological Constant Predictor
============================================================

Command-line interface for first-principles cosmological constant prediction
using the unified LQG framework.

Usage:
    python predict_cosmological_constant.py --mode=first_principles
    python predict_cosmological_constant.py --scale=1e-15 --validate

Author: LQG Cosmological Constant Predictor Team
Date: July 3, 2025
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from cosmological_constant_predictor import (
    CosmologicalConstantPredictor, 
    CosmologicalParameters
)
from cross_scale_validator import CrossScaleValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lqg_prediction.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='LQG Cosmological Constant Predictor - First-Principles Vacuum Energy Calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode=first_principles
  %(prog)s --scale=1e-15 --validate --verbose
  %(prog)s --output=prediction.json
        """
    )
    
    # Core prediction options
    parser.add_argument('--mode', choices=['first_principles', 'validation'],
                       default='first_principles',
                       help='Prediction mode (default: first_principles)')
    parser.add_argument('--scale', type=float, default=1e-15,
                       help='Target length scale in meters (default: 1e-15)')
    parser.add_argument('--scale_range', type=str,
                       help='Scale range as "min,max" in meters (e.g., "1e-35,1e26")')
    
    # Validation options
    parser.add_argument('--validate', action='store_true',
                       help='Perform cross-scale validation')
    parser.add_argument('--detailed_validation', action='store_true',
                       help='Perform detailed physics validation (slower)')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # LQG parameters
    parser.add_argument('--mu_polymer', type=float, default=0.15,
                       help='Polymer parameter μ (default: 0.15)')
    parser.add_argument('--gamma_coefficient', type=float, default=1.0,
                       help='Scale-dependent coupling γ (default: 1.0)')
    
    return parser.parse_args()

def setup_logging(args):
    """Setup logging configuration"""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

def parse_range(range_str):
    """Parse range string like '1e-35,1e26' into tuple of floats"""
    try:
        min_val, max_val = range_str.split(',')
        return float(min_val), float(max_val)
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'min,max'")

def predict_cosmological_constant(args):
    """Perform first-principles cosmological constant prediction"""
    logger.info("🎯 First-Principles Cosmological Constant Prediction")
    logger.info("=" * 53)
    
    # Setup parameters
    params = CosmologicalParameters(
        mu_polymer=args.mu_polymer,
        gamma_coefficient=args.gamma_coefficient
    )
    
    # Initialize predictor
    predictor = CosmologicalConstantPredictor(params)
    
    # Perform prediction
    result = predictor.predict_lambda_from_first_principles(
        target_scale=args.scale,
        include_uncertainty=True
    )
    
    # Display results
    print(f"\n🌌 Cosmological Constant Prediction")
    print("-" * 35)
    print(f"Target Scale:               {args.scale:.2e} m")
    print(f"Cosmological Constant:      {result.lambda_effective:.3e} m⁻²")
    print(f"Vacuum Energy Density:      {result.vacuum_energy_density:.3e} J/m³")
    print(f"Enhancement Factor:         {result.enhancement_factor:.3f}")
    print(f"Scale Correction:           {result.scale_correction:.3e}")
    print()
    
    print(f"📊 Polymer Parameters")
    print("-" * 21)
    print(f"Base μ parameter:           {result.mu_scale:.3f}")
    print(f"Base Λ₀:                    {result.lambda_0:.3e} m⁻²")
    print()
    
    print(f"🔍 Uncertainty Analysis")
    print("-" * 23)
    print(f"Prediction Uncertainty:     ±{result.lambda_uncertainty:.2e} m⁻²")
    print(f"95% Confidence Interval:    [{result.confidence_interval[0]:.2e}, {result.confidence_interval[1]:.2e}] m⁻²")
    print(f"Cross-Scale Consistency:    {result.cross_scale_consistency:.6f}")
    print()
    
    return result

def validate_cross_scale(args):
    """Perform cross-scale validation"""
    logger.info("🔍 Cross-Scale Validation")
    logger.info("=" * 25)
    
    # Setup parameters
    params = CosmologicalParameters(
        mu_polymer=args.mu_polymer,
        gamma_coefficient=args.gamma_coefficient
    )
    
    # Initialize predictor
    predictor = CosmologicalConstantPredictor(params)
    
    # Determine scale range
    if args.scale_range:
        scale_min, scale_max = parse_range(args.scale_range)
    else:
        scale_min, scale_max = 1e-35, 1e26  # Planck to cosmological
    
    # Perform validation
    validation = predictor.validate_cross_scale_consistency(
        scale_range=(scale_min, scale_max),
        num_scales=61
    )
    
    # Display results
    print(f"\n📈 Cross-Scale Validation Results")
    print("-" * 33)
    print(f"Scale Range:                {validation['scale_range_orders']:.1f} orders of magnitude")
    print(f"Scales Tested:              {validation['num_scales_tested']}")
    print(f"Consistency Score:          {validation['consistency_score']:.6f}")
    print(f"Mean Λ Value:               {validation['lambda_mean']:.3e} m⁻²")
    print(f"Standard Deviation:         {validation['lambda_std']:.3e} m⁻²")
    print(f"Relative Variation:         {validation['lambda_relative_variation']:.2e}")
    print(f"Scaling Exponent:           {validation['scaling_exponent']:.3f}")
    print()
    
    # Generate recommendations
    if validation['consistency_score'] > 0.9:
        print("✅ Excellent cross-scale consistency achieved")
    elif validation['consistency_score'] > 0.7:
        print("⚠️ Good cross-scale consistency, minor variations present")
    else:
        print("❌ Poor cross-scale consistency, review parameters")
    
    print()
    
    return validation

def save_results(results, output_file):
    """Save results to JSON file"""
    try:
        # Convert results to JSON-serializable format
        def convert_for_json(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        json_results = convert_for_json(results)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def main():
    """Main entry point"""
    args = parse_arguments()
    setup_logging(args)
    
    logger.info("🌌 LQG Cosmological Constant Predictor")
    logger.info("=====================================")
    logger.info("First-Principles Vacuum Energy Calculation")
    logger.info("")
    
    try:
        results = {}
        
        if args.mode == 'first_principles':
            prediction = predict_cosmological_constant(args)
            results['prediction'] = prediction
            
            if args.validate:
                validation = validate_cross_scale(args)
                results['validation'] = validation
                
        elif args.mode == 'validation':
            validation = validate_cross_scale(args)
            results['validation'] = validation
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
            
        logger.info("✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()
