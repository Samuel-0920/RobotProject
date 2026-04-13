
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
