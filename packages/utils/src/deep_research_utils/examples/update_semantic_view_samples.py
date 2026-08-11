"""
Example script demonstrating how to use update_semantic_view_sample_values
to enrich a semantic view configuration with sample dimension values from Snowflake.

This script reads the ECAP semantic view configuration, queries Snowflake to discover
actual dimension values, and updates the configuration with sample values for dimensions
that have 10 or fewer unique values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "utils" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core" / "src"))

from deep_research_utils.semantic_view import update_semantic_view_sample_values


def main():
    """
    Update the ECAP semantic view configuration with sample values from Snowflake.
    
    This function demonstrates the typical usage pattern:
    1. Read the original semantic view YAML
    2. Connect to Snowflake (using programmatic authentication from env vars)
    3. Query each dimension to find unique values
    4. Update sample_values for dimensions with <= 10 unique values
    5. Save the enriched configuration to a new file
    """
    
    input_yaml = "configs/ecap_semantic_view.yaml"
    output_yaml = "configs/ecap_semantic_view_with_samples.yaml"
    
    print(f"Updating semantic view sample values...")
    print(f"  Input:  {input_yaml}")
    print(f"  Output: {output_yaml}")
    print()
    
    try:
        update_semantic_view_sample_values(
            yaml_path=input_yaml,
            output_path=output_yaml,
            max_unique_values=10
        )
        
        print()
        print("✅ Successfully updated semantic view configuration!")
        print(f"   Review the updated file at: {output_yaml}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure the input YAML file exists.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Check your Snowflake credentials and connection settings.")
        sys.exit(1)


if __name__ == "__main__":
    main()
