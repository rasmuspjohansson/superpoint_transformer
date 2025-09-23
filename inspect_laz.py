# This program uses the laspy library to read a LAS file and print all unique
# classification values. It uses argparse to handle command-line arguments.

import argparse
import laspy
import numpy as np
import sys

# Standard LAS classification values and their descriptions.
# This dictionary is used as a fallback to ensure the script works
# with various laspy versions.
CLASSIFICATION_LOOKUP = {
    0: "Never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Low point (noise)",
    8: "Model Key-point (mass point)",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Overhead structure",
    13: "Wire-conductors (shield)",
    14: "Transmission Tower",
    15: "Wire-guard (shield)",
    16: "Bridge deck",
    17: "High noise",
    18: "Reserved for ASPRS"
}

def analyze_las_classifications(las_file_path):
    """
    Reads a LAS file, finds all unique classification values, and prints them.
    
    Args:
        las_file_path (str): The path to the LAS file.
    """
    try:
        # Use a 'with' statement for safe file handling
        with laspy.open(las_file_path, mode="r") as las_file:
            print(f"Analyzing file: {las_file_path}")

            # Read the classification data from the file
            las_data = las_file.read()
            
            # Get the unique classification values using numpy
            unique_classifications = np.unique(las_data.classification)
            
            # Print the results
            if unique_classifications.size > 0:
                print("\nFound the following unique classification values:")
                print("---------------------------------------------")
                
                # Iterate through the unique values and their descriptions
                for cls_value in unique_classifications:
                    # Use the hardcoded dictionary to get the description
                    # Use .get() with a default value to handle non-standard codes
                    cls_description = CLASSIFICATION_LOOKUP.get(int(cls_value), "Unknown")
                    print(f"  Value: {cls_value:<3} -> Description: {cls_description}")
            else:
                print("No classification values found in the file.")
                
    except FileNotFoundError:
        print(f"Error: The file '{las_file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse command-line arguments and run the analysis.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Analyzes a LAS file to find and print all unique classification values."
    )
    
    # Add the --lasfile argument
    parser.add_argument(
        "--lasfile",
        type=str,
        required=True,
        help="The path to the input LAS file (.las, .laz)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the analysis with the provided file path
    analyze_las_classifications(args.lasfile)

if __name__ == "__main__":
    main()
