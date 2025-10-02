import laspy
import sys

def fix_laz_georeferencing(input_file, output_file):
    """
    Fix LAZ file georeferencing by setting the WKT flag in global encoding.
    
    Args:
        input_file: Path to input .laz or .las file
        output_file: Path to output .laz or .las file
    """
    try:
        # Read the LAZ file
        print(f"Reading {input_file}...")
        las = laspy.read(input_file)
        
        # Get current point format
        point_format = las.point_format.id
        print(f"Point format: {point_format}")
        
        # Check if point format is 6-10
        if point_format >= 6 and point_format <= 10:
            # Get current global encoding
            current_encoding = las.header.global_encoding
            print(f"Current global encoding: {current_encoding}")
            
            # Set the WKT bit (bit 4) in global encoding
            # Bit 4 = 16 in decimal
            WKT_BIT = 16
            new_encoding = current_encoding | WKT_BIT
            las.header.global_encoding = new_encoding
            
            print(f"New global encoding: {new_encoding}")
            print(f"WKT flag is now set")
        else:
            print(f"Point format {point_format} doesn't require WKT flag fix")
        
        # Write the corrected file
        print(f"Writing corrected file to {output_file}...")
        las.write(output_file)
        
        print("✓ File successfully corrected!")
        print(f"You can now open {output_file} in QGIS")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    input_file = "your_input_file.laz"
    output_file = "fixed_output_file.laz"
    
    # You can also use command line arguments
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    
    fix_laz_georeferencing(input_file, output_file)
