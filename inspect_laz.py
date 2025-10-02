# This program uses the laspy library to read LAS/LAZ files and print all unique
# classification values. It accepts either a single file or a folder containing
# multiple LAS/LAZ files.

import argparse
import laspy
import numpy as np
import sys
from pathlib import Path

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
        las_file_path (str or Path): The path to the LAS file.
    
    Returns:
        int or None: Number of points in the file, or None if an error occurred.
    """
    try:
        # Use a 'with' statement for safe file handling
        with laspy.open(las_file_path, mode="r") as las_file:
            print(f"\nAnalyzing file: {las_file_path}")

            # Read the classification data from the file
            las_data = las_file.read()
            
            # Get the number of points
            num_points = len(las_data.points)
            print(f"Number of points: {num_points:,}")
            
            # Get the unique classification values using numpy
            unique_classifications = np.unique(las_data.classification)
            
            # Print the results
            if unique_classifications.size > 0:
                print("Found the following unique classification values:")
                print("---------------------------------------------")
                
                # Iterate through the unique values and their descriptions
                for cls_value in unique_classifications:
                    # Use the hardcoded dictionary to get the description
                    # Use .get() with a default value to handle non-standard codes
                    cls_description = CLASSIFICATION_LOOKUP.get(int(cls_value), "Unknown")
                    print(f"  Value: {cls_value:<3} -> Description: {cls_description}")
            else:
                print("No classification values found in the file.")
            
            return num_points
                
    except FileNotFoundError:
        print(f"Error: The file '{las_file_path}' was not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing file '{las_file_path}': {e}", file=sys.stderr)
        return None

def process_path(input_path):
    """
    Process either a single file or all LAS/LAZ files in a folder.
    
    Args:
        input_path (str): Path to either a file or folder.
    """
    path = Path(input_path)
    
    if not path.exists():
        print(f"Error: The path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if path.is_file():
        # Process single file
        if path.suffix.lower() in ['.las', '.laz']:
            analyze_las_classifications(path)
        else:
            print(f"Error: '{input_path}' is not a LAS or LAZ file.", file=sys.stderr)
            sys.exit(1)
    
    elif path.is_dir():
        # Process all LAS/LAZ files in the folder
        las_files = list(path.glob('*.las')) + list(path.glob('*.laz'))
        las_files += list(path.glob('*.LAS')) + list(path.glob('*.LAZ'))
        
        if not las_files:
            print(f"No LAS or LAZ files found in '{input_path}'.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(las_files)} LAS/LAZ file(s) in '{input_path}'")
        print("=" * 60)
        
        # Store file results for summary
        file_results = []
        
        for las_file in sorted(las_files):
            num_points = analyze_las_classifications(las_file)
            if num_points is not None:
                file_results.append((las_file.name, num_points))
        
        # Print summary sorted by number of points (ascending)
        if file_results:
            print("\n" + "=" * 60)
            print("SUMMARY - Files sorted by number of points:")
            print("=" * 60)
            
            # Sort by number of points (ascending - smallest first, largest last)
            file_results.sort(key=lambda x: x[1])
            
            # Calculate the maximum filename length for alignment
            max_name_len = max(len(name) for name, _ in file_results)
            
            for filename, num_points in file_results:
                print(f"  {filename:<{max_name_len}} : {num_points:>15,} points")
            
            # Print total
            total_points = sum(num_points for _, num_points in file_results)
            print("-" * 60)
            print(f"  {'TOTAL':<{max_name_len}} : {total_points:>15,} points")
        
        print("\n" + "=" * 60)
        print(f"Completed analysis of {len(file_results)} file(s).")
    
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse command-line arguments and run the analysis.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Analyzes LAS/LAZ file(s) to find and print all unique classification values. "
                    "Accepts either a single file or a folder containing multiple files."
    )
    
    # Add the --input argument (renamed from --lasfile for clarity)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to either a single LAS/LAZ file or a folder containing LAS/LAZ files."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the analysis with the provided path
    process_path(args.input)

if __name__ == "__main__":
    main()
