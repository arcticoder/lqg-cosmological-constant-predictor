#!/usr/bin/env python3
"""
Warp Drive Designer - Interactive Design Tool
=============================================

Interactive command-line tool for designing warp drive systems using
first-principles cosmological constant predictions.

Usage:
    python warp_drive_designer.py --velocity=0.1 --radius=100
    python warp_drive_designer.py --interactive
    python warp_drive_designer.py --preset=explorer --optimize

Author: LQG Cosmological Constant Predictor Team
Date: July 3, 2025
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from cosmological_constant_predictor import CosmologicalConstantPredictor
from warp_drive_engineer import WarpDriveEngineer, WarpBubbleSpecification
from cross_scale_validator import CrossScaleValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Design presets for common applications
DESIGN_PRESETS = {
    'probe': {
        'velocity': 0.01,  # 1% light speed
        'radius': 10.0,    # 10 meter probe
        'description': 'Interplanetary probe design'
    },
    'courier': {
        'velocity': 0.05,  # 5% light speed
        'radius': 50.0,    # 50 meter courier ship
        'description': 'Fast courier vessel'
    },
    'explorer': {
        'velocity': 0.1,   # 10% light speed
        'radius': 100.0,   # 100 meter exploration vessel
        'description': 'Interstellar exploration ship'
    },
    'transport': {
        'velocity': 0.2,   # 20% light speed
        'radius': 500.0,   # 500 meter transport
        'description': 'Large cargo/passenger transport'
    },
    'research': {
        'velocity': 0.05,  # 5% light speed
        'radius': 200.0,   # 200 meter research station
        'description': 'Mobile research platform'
    }
}

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Warp Drive Designer - Interactive Design Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Design Presets:
  probe     - Small interplanetary probe (1%c, 10m)
  courier   - Fast courier vessel (5%c, 50m)
  explorer  - Exploration ship (10%c, 100m)
  transport - Large transport (20%c, 500m)
  research  - Research platform (5%c, 200m)

Examples:
  %(prog)s --velocity=0.1 --radius=100
  %(prog)s --preset=explorer --optimize
  %(prog)s --interactive
  %(prog)s --preset=transport --lab_config --output=transport_design.json
        """
    )
    
    # Design parameters
    parser.add_argument('--velocity', type=float,
                       help='Warp velocity as fraction of c (0 < v < 1)')
    parser.add_argument('--radius', type=float,
                       help='Warp bubble radius in meters')
    parser.add_argument('--preset', choices=DESIGN_PRESETS.keys(),
                       help='Use design preset')
    
    # Operation modes
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive design mode')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize the design')
    parser.add_argument('--lab_config', action='store_true',
                       help='Generate laboratory configuration')
    parser.add_argument('--blueprint', action='store_true',
                       help='Generate complete engineering blueprint')
    
    # Optimization parameters
    parser.add_argument('--optimization_target', choices=['energy', 'feasibility', 'efficiency'],
                       default='feasibility',
                       help='Optimization target (default: feasibility)')
    parser.add_argument('--velocity_range', type=str, default='0.01,0.3',
                       help='Velocity range for optimization (default: "0.01,0.3")')
    parser.add_argument('--radius_range', type=str, default='10,1000',
                       help='Radius range for optimization (default: "10,1000")')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for design results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--validate', action='store_true',
                       help='Validate design with cross-scale analysis')
    
    return parser.parse_args()

def interactive_design():
    """Interactive design interface"""
    print("🚀 Interactive Warp Drive Designer")
    print("=" * 34)
    print()
    
    # Show available presets
    print("📋 Available Design Presets:")
    print("-" * 27)
    for name, preset in DESIGN_PRESETS.items():
        print(f"{name:10s}: {preset['description']} ({preset['velocity']:.1%}c, {preset['radius']:.0f}m)")
    print()
    
    # Get design parameters
    while True:
        choice = input("Choose preset or enter 'custom' for manual design: ").strip().lower()
        
        if choice == 'custom':
            # Manual parameter input
            while True:
                try:
                    velocity = float(input("Enter warp velocity (as fraction of c, e.g., 0.1 for 10%): "))
                    if 0 < velocity < 1:
                        break
                    else:
                        print("Velocity must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")
            
            while True:
                try:
                    radius = float(input("Enter bubble radius (meters): "))
                    if radius > 0:
                        break
                    else:
                        print("Radius must be positive")
                except ValueError:
                    print("Please enter a valid number")
            
            break
            
        elif choice in DESIGN_PRESETS:
            preset = DESIGN_PRESETS[choice]
            velocity = preset['velocity']
            radius = preset['radius']
            print(f"\nSelected preset: {preset['description']}")
            print(f"Velocity: {velocity:.1%}c, Radius: {radius:.0f}m")
            break
        else:
            print("Invalid choice. Please select a preset or enter 'custom'")
    
    print()
    
    # Design options
    optimize = input("Optimize design? (y/n): ").strip().lower() == 'y'
    lab_config = input("Generate laboratory configuration? (y/n): ").strip().lower() == 'y'
    blueprint = input("Generate engineering blueprint? (y/n): ").strip().lower() == 'y'
    validate = input("Perform cross-scale validation? (y/n): ").strip().lower() == 'y'
    
    return {
        'velocity': velocity,
        'radius': radius,
        'optimize': optimize,
        'lab_config': lab_config,
        'blueprint': blueprint,
        'validate': validate
    }

def parse_range(range_str):
    """Parse range string into tuple of floats"""
    try:
        min_val, max_val = map(float, range_str.split(','))
        return (min_val, max_val)
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Use 'min,max' format.")

def design_warp_drive(velocity, radius, args):
    """Design warp drive system"""
    print(f"🎯 Designing Warp Drive System")
    print("-" * 30)
    print(f"Target Velocity: {velocity:.1%}c")
    print(f"Bubble Radius:   {radius:.1f} m")
    print()
    
    # Initialize engineer
    engineer = WarpDriveEngineer()
    
    # Generate design
    specification = engineer.design_warp_bubble(velocity, radius)
    
    # Display main specifications
    print(f"⚡ Core Specifications")
    print("-" * 21)
    print(f"Wall Thickness:             {specification.wall_thickness:.2e} m")
    print(f"Total Energy Required:      {specification.total_energy:.2e} J")
    print(f"Power Requirement:          {specification.power_requirement:.2e} W")
    print(f"Electric Field Strength:    {specification.electric_field_strength:.2e} V/m")
    print(f"Magnetic Field Strength:    {specification.magnetic_field_strength:.2e} T")
    print(f"Field Configuration:        {specification.field_configuration}")
    print()
    
    print(f"🧱 Material Requirements")
    print("-" * 24)
    print(f"Exotic Matter Mass:         {specification.exotic_matter_mass:.2e} kg")
    print(f"Matter Density:             {specification.matter_density:.2e} kg/m³")
    print(f"Casimir Plates Needed:      {specification.casimir_plates:,}")
    print(f"Plate Separation:           {specification.plate_separation:.2e} m")
    print(f"Total Casimir Force:        {specification.casimir_force:.2e} N")
    print()
    
    print(f"📊 Performance Metrics")
    print("-" * 22)
    print(f"Maximum Acceleration:       {specification.acceleration_limit:.2e} m/s²")
    print(f"Energy Efficiency:          {specification.efficiency:.3f}")
    print(f"Feasibility Score:          {specification.feasibility_score:.3f}")
    print()
    
    # Assessment
    if specification.feasibility_score > 0.7:
        assessment = "🟢 HIGHLY FEASIBLE"
    elif specification.feasibility_score > 0.4:
        assessment = "🟡 MODERATELY FEASIBLE"
    elif specification.feasibility_score > 0.1:
        assessment = "🟠 CHALLENGING"
    else:
        assessment = "🔴 EXTREMELY CHALLENGING"
    
    print(f"🎯 Feasibility Assessment: {assessment}")
    print()
    
    return specification

def optimize_design(specification, args):
    """Optimize warp drive design"""
    print(f"⚡ Design Optimization")
    print("-" * 21)
    
    engineer = WarpDriveEngineer()
    
    # Parse parameter ranges
    velocity_range = parse_range(args.velocity_range)
    radius_range = parse_range(args.radius_range)
    
    print(f"Velocity Range: {velocity_range[0]:.1%}c - {velocity_range[1]:.1%}c")
    print(f"Radius Range:   {radius_range[0]:.0f}m - {radius_range[1]:.0f}m")
    print(f"Target:         {args.optimization_target}")
    print()
    
    # Perform optimization
    optimization = engineer.optimize_bubble_design(
        velocity_range=velocity_range,
        radius_range=radius_range,
        optimization_target=args.optimization_target
    )
    
    if optimization['best_specification']:
        best = optimization['best_specification']
        
        print(f"🏆 Optimization Results")
        print("-" * 23)
        print(f"Optimal Velocity:           {optimization['best_velocity']:.1%}c")
        print(f"Optimal Radius:             {optimization['best_radius']:.1f} m")
        print(f"Optimization Score:         {optimization['best_score']:.3f}")
        print()
        
        print(f"📈 Improved Performance")
        print("-" * 23)
        
        # Compare with original
        if hasattr(specification, 'total_energy'):
            energy_improvement = specification.total_energy / best.total_energy
            print(f"Energy Improvement:         {energy_improvement:.1f}×")
        
        efficiency_improvement = best.efficiency / specification.efficiency
        feasibility_improvement = best.feasibility_score / specification.feasibility_score
        
        print(f"Efficiency Improvement:     {efficiency_improvement:.1f}×")
        print(f"Feasibility Improvement:    {feasibility_improvement:.1f}×")
        print()
        
        print(f"📊 Optimization Statistics")
        print("-" * 26)
        print(f"Designs Evaluated:          {len(optimization['all_results'])}")
        print(f"Success Rate:               {len([r for r in optimization['all_results'] if r['score'] > 0])}/{len(optimization['all_results'])}")
        print()
        
        return optimization['best_specification']
    else:
        print("❌ Optimization failed - no improved designs found")
        return specification

def generate_lab_configuration(specification, args):
    """Generate laboratory configuration"""
    print(f"🔬 Laboratory Configuration")
    print("-" * 27)
    
    engineer = WarpDriveEngineer()
    
    # Ask for scale factor in interactive mode
    if hasattr(args, 'interactive') and args.interactive:
        print("Scale factors for laboratory testing:")
        print("  nano-scale (1e-9):  Nanofabrication, highest precision")
        print("  micro-scale (1e-6): Microfabrication, good precision")
        print("  milli-scale (1e-3): Mesoscale, easier fabrication")
        
        while True:
            try:
                scale_input = input("Enter scale factor (e.g., 1e-9) or choose nano/micro/milli: ").strip().lower()
                
                if scale_input == 'nano':
                    scale_factor = 1e-9
                elif scale_input == 'micro':
                    scale_factor = 1e-6
                elif scale_input == 'milli':
                    scale_factor = 1e-3
                else:
                    scale_factor = float(scale_input)
                
                if 0 < scale_factor <= 1:
                    break
                else:
                    print("Scale factor must be between 0 and 1")
            except ValueError:
                print("Please enter a valid scale factor")
    else:
        scale_factor = 1e-9  # Default nano-scale
    
    lab_config = engineer.design_laboratory_configuration(specification, scale_factor)
    
    print(f"Scale Factor:               {scale_factor:.0e}")
    print()
    
    print(f"🏗️ Physical Setup")
    print("-" * 17)
    dims = lab_config.casimir_cavity_dimensions
    print(f"Cavity Dimensions:          {dims[0]:.2e} × {dims[1]:.2e} × {dims[2]:.2e} m")
    print(f"Cavity Volume:              {dims[0] * dims[1] * dims[2]:.2e} m³")
    print(f"Plate Material:             {lab_config.casimir_plate_material}")
    print(f"Plate Count:                {lab_config.casimir_plate_count}")
    print()
    
    print(f"⚡ Field Generation")
    print("-" * 18)
    coil_specs = lab_config.superconducting_coil_specs
    print(f"Coil Radius:                {coil_specs['radius']:.2e} m")
    print(f"Coil Current:               {coil_specs['current']:.2e} A")
    print(f"Coil Inductance:            {coil_specs['inductance']:.2e} H")
    print(f"Capacitor Energy:           {lab_config.capacitor_bank_energy:.2e} J")
    print(f"Pulse Duration:             {lab_config.pulse_duration:.2e} s")
    print()
    
    print(f"🛡️ Safety Systems")
    print("-" * 17)
    print(f"Containment Field:          {lab_config.containment_field_strength:.2e} T")
    print(f"Emergency Shutdown:         {lab_config.emergency_shutdown_time:.2e} s")
    print(f"Radiation Shielding:        {lab_config.radiation_shielding_thickness:.1f} m")
    print()
    
    if lab_config.metamaterial_type != "none":
        print(f"🔬 Metamaterial Enhancement")
        print("-" * 27)
        print(f"Metamaterial Type:          {lab_config.metamaterial_type}")
        print(f"Metamaterial Volume:        {lab_config.metamaterial_volume:.2e} m³")
        print()
    
    return lab_config

def generate_blueprint(specification, lab_config, args):
    """Generate engineering blueprint"""
    print(f"📋 Engineering Blueprint")
    print("-" * 24)
    
    engineer = WarpDriveEngineer()
    blueprint = engineer.generate_engineering_blueprint(specification, lab_config)
    
    print(f"📄 Project Overview")
    print("-" * 19)
    overview = blueprint['project_overview']
    print(f"Title:                      {overview['title']}")
    print(f"Velocity Target:            {overview['velocity_target']}")
    print(f"Bubble Radius:              {overview['bubble_radius']}")
    print(f"Energy Requirement:         {overview['energy_requirement']}")
    print(f"Feasibility Score:          {overview['feasibility_score']}")
    print()
    
    print(f"🏗️ Structural Systems")
    print("-" * 22)
    structure = blueprint['structural_specifications']
    geometry = structure['bubble_geometry']
    print(f"Outer Radius:               {geometry['outer_radius']:.1f} m")
    print(f"Wall Thickness:             {geometry['wall_thickness']:.2e} m")
    print(f"Primary Material:           {geometry['material']}")
    print(f"Structural Strength:        {geometry['structural_strength']}")
    print()
    
    print(f"⚡ Field Generation")
    print("-" * 18)
    fields = blueprint['field_generation_systems']['primary_field_array']
    print(f"Field Type:                 {fields['type']}")
    print(f"Electric Field:             {fields['electric_field_strength']}")
    print(f"Magnetic Field:             {fields['magnetic_field_strength']}")
    print(f"Power Requirement:          {fields['power_requirement']}")
    print()
    
    print(f"🧱 Exotic Matter Systems")
    print("-" * 24)
    matter = blueprint['exotic_matter_systems']
    print(f"Total Requirement:          {matter['total_requirement']}")
    print(f"Density Requirement:        {matter['density_requirement']}")
    print(f"Generation Method:          {matter['generation_method']}")
    print(f"Containment Field:          {matter['containment_field']}")
    print()
    
    print(f"🎮 Control Systems")
    print("-" * 18)
    control = blueprint['control_systems']
    field_control = control['field_control']
    nav_system = control['navigation_system']
    print(f"Response Time:              {field_control['response_time']}")
    print(f"Stability Requirement:     {field_control['stability_requirement']}")
    print(f"Max Acceleration:           {nav_system['acceleration_limit']}")
    print(f"Position Accuracy:          {nav_system['position_accuracy']}")
    print()
    
    print(f"📈 Performance Predictions")
    print("-" * 26)
    performance = blueprint['performance_predictions']
    print(f"Efficiency:                 {performance['efficiency']}")
    print(f"Energy Cost per Trip:       {performance['energy_cost_per_trip']}")
    print(f"Maximum Range:              {performance['maximum_range']}")
    print()
    
    return blueprint

def perform_validation(specification, args):
    """Perform cross-scale validation"""
    print(f"🔍 Cross-Scale Validation")
    print("-" * 25)
    
    validator = CrossScaleValidator()
    validation = validator.validate_across_scales(detailed_analysis=True)
    report = validator.generate_validation_report(validation)
    
    print(f"Overall Consistency:        {validation.overall_consistency:.3f}")
    print(f"Scale Coverage:             {validation.scale_coverage:.1f} orders of magnitude")
    print(f"Validation Status:          {report['executive_summary']['validation_status']}")
    print()
    
    if validation.overall_consistency > 0.8:
        print("✅ Design validated across all physical scales")
    elif validation.overall_consistency > 0.6:
        print("⚠️  Design validated with minor inconsistencies")
    else:
        print("❌ Design requires validation review")
    
    print()
    
    return validation, report

def save_design(results, output_file):
    """Save design results to JSON file"""
    try:
        # Convert complex objects to JSON-serializable format
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
        
        print(f"💾 Design saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to save design: {e}")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    print("🚀 Warp Drive Designer")
    print("=" * 22)
    print("First-Principles Engineering Tool")
    print()
    
    results = {}
    
    try:
        # Get design parameters
        if args.interactive:
            params = interactive_design()
            velocity = params['velocity']
            radius = params['radius']
            # Override args with interactive choices
            args.optimize = params['optimize']
            args.lab_config = params['lab_config']
            args.blueprint = params['blueprint']
            args.validate = params['validate']
        elif args.preset:
            preset = DESIGN_PRESETS[args.preset]
            velocity = preset['velocity']
            radius = preset['radius']
            print(f"Using preset: {preset['description']}")
            print(f"Velocity: {velocity:.1%}c, Radius: {radius:.0f}m")
            print()
        elif args.velocity and args.radius:
            velocity = args.velocity
            radius = args.radius
        else:
            print("❌ Error: Must specify --velocity and --radius, use --preset, or use --interactive mode")
            sys.exit(1)
        
        # Validate parameters
        if not (0 < velocity < 1):
            print(f"❌ Error: Velocity must be between 0 and 1, got {velocity}")
            sys.exit(1)
        
        if radius <= 0:
            print(f"❌ Error: Radius must be positive, got {radius}")
            sys.exit(1)
        
        # Design warp drive
        specification = design_warp_drive(velocity, radius, args)
        results['specification'] = specification
        
        # Optimization
        if args.optimize:
            print("\n" + "="*50)
            optimized_spec = optimize_design(specification, args)
            results['optimized_specification'] = optimized_spec
            specification = optimized_spec  # Use optimized version for subsequent steps
        
        # Laboratory configuration
        if args.lab_config:
            print("\n" + "="*50)
            lab_config = generate_lab_configuration(specification, args)
            results['lab_configuration'] = lab_config
        else:
            lab_config = None
        
        # Engineering blueprint
        if args.blueprint:
            print("\n" + "="*50)
            if lab_config is None:
                # Generate lab config if not done already
                engineer = WarpDriveEngineer()
                lab_config = engineer.design_laboratory_configuration(specification)
                results['lab_configuration'] = lab_config
            
            blueprint = generate_blueprint(specification, lab_config, args)
            results['blueprint'] = blueprint
        
        # Validation
        if args.validate:
            print("\n" + "="*50)
            validation, report = perform_validation(specification, args)
            results['validation'] = validation
            results['validation_report'] = report
        
        # Save results
        if args.output:
            save_design(results, args.output)
        
        print("\n" + "="*60)
        print("✅ Warp Drive Design Complete!")
        print("   Ready for precision exotic matter engineering.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Design cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Design failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
