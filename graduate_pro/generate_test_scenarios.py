#!/usr/bin/env python3
# generate_test_scenarios.py - Generate 24 test scenario metadata files

import json
import os
from pathlib import Path
from typing import List, Dict

def generate_controlled_scenarios() -> List[Dict]:
    """Generate 12 controlled test scenarios"""
    
    controlled_scenarios = [
        # Normal lighting scenarios
        {
            "scenario_id": "controlled_001_normal_single",
            "complexity_level": 1,
            "lighting_condition": "normal",
            "object_count": 1,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_simple",
            "notes": "Single target object under normal lighting, clean background"
        },
        {
            "scenario_id": "controlled_002_normal_multi",
            "complexity_level": 2,
            "lighting_condition": "normal",
            "object_count": 4,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_multi",
            "notes": "Multiple target objects under normal lighting, well separated"
        },
        
        # Dim lighting scenarios
        {
            "scenario_id": "controlled_003_dim_single",
            "complexity_level": 2,
            "lighting_condition": "dim",
            "object_count": 1,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_simple",
            "notes": "Single target object under dim lighting conditions"
        },
        {
            "scenario_id": "controlled_004_dim_multi",
            "complexity_level": 3,
            "lighting_condition": "dim",
            "object_count": 4,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_multi",
            "notes": "Multiple target objects under dim lighting, increased difficulty"
        },
        
        # Backlight scenarios
        {
            "scenario_id": "controlled_005_backlit_single",
            "complexity_level": 3,
            "lighting_condition": "backlit",
            "object_count": 1,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_challenging",
            "notes": "Single target with strong backlight creating silhouettes"
        },
        {
            "scenario_id": "controlled_006_backlit_multi",
            "complexity_level": 4,
            "lighting_condition": "backlit",
            "object_count": 3,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "controlled_challenging",
            "notes": "Multiple targets with challenging backlight conditions"
        },
        
        # Interference object scenarios
        {
            "scenario_id": "controlled_007_interference_single",
            "complexity_level": 3,
            "lighting_condition": "normal",
            "object_count": 1,
            "interference_objects": 1,
            "occlusion_level": 0.0,
            "scene_type": "controlled_interference",
            "notes": "Single target with one similar-looking interference object"
        },
        {
            "scenario_id": "controlled_008_interference_multi",
            "complexity_level": 4,
            "lighting_condition": "normal",
            "object_count": 3,
            "interference_objects": 3,
            "occlusion_level": 0.0,
            "scene_type": "controlled_interference",
            "notes": "Multiple targets with multiple interference objects of similar appearance"
        },
        
        # Occlusion scenarios
        {
            "scenario_id": "controlled_009_occlusion_light",
            "complexity_level": 3,
            "lighting_condition": "normal",
            "object_count": 2,
            "interference_objects": 0,
            "occlusion_level": 0.3,
            "scene_type": "controlled_occlusion",
            "notes": "Targets with 30% occlusion using transparent/semi-transparent materials"
        },
        {
            "scenario_id": "controlled_010_occlusion_moderate",
            "complexity_level": 4,
            "lighting_condition": "normal",
            "object_count": 3,
            "interference_objects": 0,
            "occlusion_level": 0.5,
            "scene_type": "controlled_occlusion",
            "notes": "Multiple targets with moderate occlusion (50%) using opaque blockers"
        },
        
        # Combined challenge scenarios
        {
            "scenario_id": "controlled_011_dim_interference",
            "complexity_level": 4,
            "lighting_condition": "dim",
            "object_count": 2,
            "interference_objects": 2,
            "occlusion_level": 0.0,
            "scene_type": "controlled_combined",
            "notes": "Dim lighting combined with interference objects"
        },
        {
            "scenario_id": "controlled_012_backlit_occlusion",
            "complexity_level": 5,
            "lighting_condition": "backlit",
            "object_count": 2,
            "interference_objects": 0,
            "occlusion_level": 0.3,
            "scene_type": "controlled_combined",
            "notes": "Most challenging: backlight with partial occlusion"
        }
    ]
    
    return controlled_scenarios

def generate_real_cluttered_scenarios() -> List[Dict]:
    """Generate 12 real cluttered test scenarios"""
    
    real_scenarios = [
        # Desktop clutter scenarios
        {
            "scenario_id": "real_001_desktop_simple",
            "complexity_level": 2,
            "lighting_condition": "normal",
            "object_count": 3,
            "interference_objects": 1,
            "occlusion_level": 0.1,
            "scene_type": "desktop_clutter",
            "notes": "Typical office desktop with papers, pens, and target objects"
        },
        {
            "scenario_id": "real_002_desktop_busy",
            "complexity_level": 4,
            "lighting_condition": "normal",
            "object_count": 5,
            "interference_objects": 3,
            "occlusion_level": 0.3,
            "scene_type": "desktop_clutter",
            "notes": "Busy desktop environment with many overlapping objects"
        },
        
        # Reflective materials
        {
            "scenario_id": "real_003_reflective_metal",
            "complexity_level": 3,
            "lighting_condition": "bright",
            "object_count": 2,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "reflective_surfaces",
            "notes": "Metallic objects creating reflections and specular highlights"
        },
        {
            "scenario_id": "real_004_reflective_glass",
            "complexity_level": 4,
            "lighting_condition": "bright",
            "object_count": 3,
            "interference_objects": 1,
            "occlusion_level": 0.2,
            "scene_type": "reflective_surfaces",
            "notes": "Glass and plastic objects with challenging reflective properties"
        },
        
        # Patterned backgrounds
        {
            "scenario_id": "real_005_patterned_fabric",
            "complexity_level": 3,
            "lighting_condition": "normal",
            "object_count": 4,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "patterned_background",
            "notes": "Objects placed on patterned fabric or textured surfaces"
        },
        {
            "scenario_id": "real_006_patterned_complex",
            "complexity_level": 5,
            "lighting_condition": "normal",
            "object_count": 3,
            "interference_objects": 2,
            "occlusion_level": 0.4,
            "scene_type": "patterned_background",
            "notes": "Complex patterns with camouflaging effects on target detection"
        },
        
        # Distance and angle variations
        {
            "scenario_id": "real_007_distance_close",
            "complexity_level": 2,
            "lighting_condition": "normal",
            "object_count": 2,
            "interference_objects": 0,
            "occlusion_level": 0.0,
            "scene_type": "distance_variation",
            "notes": "Close-up view (< 30cm) with high detail but potential distortion"
        },
        {
            "scenario_id": "real_008_distance_far",
            "complexity_level": 3,
            "lighting_condition": "normal",
            "object_count": 4,
            "interference_objects": 1,
            "occlusion_level": 0.0,
            "scene_type": "distance_variation",
            "notes": "Far view (> 60cm) with multiple small objects"
        },
        
        # Angle variations
        {
            "scenario_id": "real_009_angle_oblique",
            "complexity_level": 3,
            "lighting_condition": "normal",
            "object_count": 3,
            "interference_objects": 1,
            "occlusion_level": 0.2,
            "scene_type": "angle_variation",
            "notes": "Oblique viewing angle (45 degrees) creating perspective challenges"
        },
        {
            "scenario_id": "real_010_angle_extreme",
            "complexity_level": 4,
            "lighting_condition": "dim",
            "object_count": 2,
            "interference_objects": 0,
            "occlusion_level": 0.1,
            "scene_type": "angle_variation",
            "notes": "Extreme viewing angle with dim lighting combination"
        },
        
        # Mixed material scenarios
        {
            "scenario_id": "real_011_mixed_textures",
            "complexity_level": 4,
            "lighting_condition": "normal",
            "object_count": 5,
            "interference_objects": 2,
            "occlusion_level": 0.3,
            "scene_type": "mixed_materials",
            "notes": "Combination of different material types: plastic, metal, fabric, wood"
        },
        {
            "scenario_id": "real_012_kitchen_scene",
            "complexity_level": 5,
            "lighting_condition": "bright",
            "object_count": 4,
            "interference_objects": 3,
            "occlusion_level": 0.4,
            "scene_type": "kitchen_environment",
            "notes": "Real kitchen counter with utensils, containers, and food items"
        }
    ]
    
    return real_scenarios

def create_metadata_files(output_dir: str = "test_scenarios_metadata"):
    """Create all 24 test scenario metadata files"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate scenarios
    controlled_scenarios = generate_controlled_scenarios()
    real_scenarios = generate_real_cluttered_scenarios()
    
    all_scenarios = controlled_scenarios + real_scenarios
    
    print(f"Generating {len(all_scenarios)} test scenario metadata files...")
    print("=" * 60)
    
    # Create individual metadata files
    for i, scenario in enumerate(all_scenarios, 1):
        filename = f"{scenario['scenario_id']}_metadata.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
        
        print(f"{i:2d}. {scenario['scenario_id']}")
        print(f"    Type: {scenario['scene_type']}")
        print(f"    Complexity: {scenario['complexity_level']}/5")
        print(f"    Lighting: {scenario['lighting_condition']}")
        print(f"    Objects: {scenario['object_count']} + {scenario['interference_objects']} interference")
        print(f"    Notes: {scenario['notes']}")
        print()
    
    # Create summary file
    summary = {
        "total_scenarios": len(all_scenarios),
        "controlled_scenarios": len(controlled_scenarios),
        "real_cluttered_scenarios": len(real_scenarios),
        "complexity_distribution": {
            "level_1": len([s for s in all_scenarios if s['complexity_level'] == 1]),
            "level_2": len([s for s in all_scenarios if s['complexity_level'] == 2]),
            "level_3": len([s for s in all_scenarios if s['complexity_level'] == 3]),
            "level_4": len([s for s in all_scenarios if s['complexity_level'] == 4]),
            "level_5": len([s for s in all_scenarios if s['complexity_level'] == 5])
        },
        "lighting_distribution": {
            "normal": len([s for s in all_scenarios if s['lighting_condition'] == 'normal']),
            "dim": len([s for s in all_scenarios if s['lighting_condition'] == 'dim']),
            "bright": len([s for s in all_scenarios if s['lighting_condition'] == 'bright']),
            "backlit": len([s for s in all_scenarios if s['lighting_condition'] == 'backlit'])
        },
        "scene_types": list(set([s['scene_type'] for s in all_scenarios])),
        "scenarios": all_scenarios
    }
    
    summary_file = output_path / "test_scenarios_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Create instruction file
    instructions = """
# Test Scenario Setup Instructions

## Overview
This folder contains metadata for 24 carefully designed test scenarios:
- 12 Controlled scenarios (testing specific conditions)
- 12 Real cluttered scenarios (testing practical situations)

## Setup Process

1. **For each scenario:**
   - Read the metadata file (e.g., controlled_001_normal_single_metadata.json)
   - Set up the physical scene according to the specifications:
     - Object count: Number of target objects to place
     - Interference objects: Number of similar-looking distractors
     - Lighting condition: Adjust room lighting accordingly
     - Occlusion level: Use transparent/opaque blockers as needed

2. **Perform the scan:**
   - Use your normal scanning procedure
   - Save results to a directory named with timestamp (e.g., scan_output_20250821_120000)

3. **Add metadata:**
   - Copy the corresponding metadata file into the scan directory
   - Rename it to "test_metadata.json"

4. **Run batch testing:**
   - Use the batch_test_runner.py to process all scans
   - The system will automatically find and use the metadata

## Lighting Conditions Guide
- **normal**: Standard room lighting, good visibility
- **dim**: Reduced lighting, shadows present but objects still visible  
- **bright**: Strong lighting, potential for overexposure
- **backlit**: Strong light source behind objects, creating silhouettes

## Complexity Levels
- **Level 1**: Single object, ideal conditions
- **Level 2**: Simple multi-object or single challenging condition
- **Level 3**: Moderate complexity, 1-2 challenging factors
- **Level 4**: High complexity, multiple challenging factors
- **Level 5**: Maximum difficulty, extreme conditions

## Expected Time Investment
- Setup per scenario: 2-3 minutes
- Scanning per scenario: 1-2 minutes  
- Total for 24 scenarios: ~2 hours setup + scanning time
"""
    
    instructions_file = output_path / "SETUP_INSTRUCTIONS.md"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("=" * 60)
    print(f"✅ Generated {len(all_scenarios)} test scenario files")
    print(f"📁 Files saved to: {output_path}")
    print(f"📋 Summary saved to: {summary_file}")
    print(f"📖 Instructions saved to: {instructions_file}")
    
    return output_path

def print_scenario_distribution(scenarios: List[Dict]):
    """Print distribution analysis"""
    print("\nScenario Distribution Analysis:")
    print("-" * 40)
    
    # Complexity distribution
    complexity_counts = {}
    for s in scenarios:
        level = s['complexity_level']
        complexity_counts[level] = complexity_counts.get(level, 0) + 1
    
    print("Complexity Levels:")
    for level in sorted(complexity_counts.keys()):
        count = complexity_counts[level]
        print(f"  Level {level}: {count} scenarios ({count/len(scenarios)*100:.1f}%)")
    
    # Lighting distribution
    lighting_counts = {}
    for s in scenarios:
        condition = s['lighting_condition']
        lighting_counts[condition] = lighting_counts.get(condition, 0) + 1
    
    print("\nLighting Conditions:")
    for condition, count in lighting_counts.items():
        print(f"  {condition}: {count} scenarios ({count/len(scenarios)*100:.1f}%)")

def main():
    print("Test Scenario Metadata Generator")
    print("=" * 60)
    
    # Generate all scenarios
    controlled = generate_controlled_scenarios()
    real = generate_real_cluttered_scenarios()
    all_scenarios = controlled + real
    
    # Print distribution
    print_scenario_distribution(all_scenarios)
    
    # Create files
    output_dir = create_metadata_files()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Review the generated scenarios in: {output_dir}")
    print(f"2. Set up each physical scene according to the metadata")
    print(f"3. Perform scans and copy metadata files to scan directories")
    print(f"4. Run batch testing with: python batch_test_runner.py /path/to/test_data")

if __name__ == '__main__':
    main()