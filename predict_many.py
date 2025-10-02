#!/usr/bin/env python
"""
Semantic Segmentation Prediction Script
Processes LAS/LAZ files using pretrained Superpoint Transformer model
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import laspy
from pyproj import CRS

# Add project path
file_path = os.path.dirname(os.path.abspath(''))
sys.path.append("/home/rajoh/projects/superpoint_transformer")

from src.transforms import SampleRecursiveMainXYAxisTiling, GridSampling3D, SampleXYTiling, NAGRemoveKeys
from src.data import Data, Batch
from src.utils.color import to_float_rgb
from src.utils import init_config
import hydra


# ============================================================================
# Dataset Configuration
# ============================================================================

# Vancouver dataset configuration (default)
VANCOUVER_NUM_CLASSES = 6

ID2TRAINID = np.asarray([
    6,  # 0 Not used         -> 6 Ignored
    5,  # 1 Other            -> 5 Other
    0,  # 2 Ground           -> 0 Ground
    3,  # 3 Low vegetation   -> 3 Low vegetation
    6,  # 4 Unknown / Noise  -> 6 Ignored
    2,  # 5 High vegetation  -> 2 High vegetation
    4,  # 6 Building         -> 4 Buildings
    6,  # 7 Unknown / Noise  -> 6 Ignored
    6,  # 8 Unknown / Noise  -> 6 Ignored
    1   # 9 Water            -> 1 Water
])

VANCOUVER_CLASS_NAMES = [
    'Ground',
    'Water',
    'High vegetation',
    'Low vegetation',
    'Buildings',
    'Other',
    'Ignored'
]

VANCOUVER_CLASS_COLORS = np.asarray([
    [243, 214, 171],
    [169, 222, 249],
    [ 70, 115,  66],
    [204, 213, 174],
    [214,  66,  54],
    [186, 160, 164],
    [  0,   0,   0]
])


# ============================================================================
# Helper Functions
# ============================================================================

def get_input_files(input_path):
    """
    Get list of LAS/LAZ files from input path.
    
    Args:
        input_path: Path to a file or directory
        
    Returns:
        List of Path objects pointing to LAS/LAZ files
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.las', '.laz']:
            return [input_path]
        else:
            raise ValueError(f"Input file must be .las or .laz, got: {input_path.suffix}")
    
    elif input_path.is_dir():
        las_files = list(input_path.glob('*.las')) + list(input_path.glob('*.laz'))
        las_files += list(input_path.glob('*.LAS')) + list(input_path.glob('*.LAZ'))
        
        if not las_files:
            raise ValueError(f"No .las or .laz files found in directory: {input_path}")
        
        return sorted(las_files)
    
    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


def read_vancouver_tile(
        filepath,
        xyz=True,
        rgb=False,
        intensity=True,
        semantic=True,
        instance=False,
        remap=True,
        max_intensity=600):
    """Read a Vancouver/LAS tile and return a Data object."""
    data = Data()
    las = laspy.read(filepath)

    print(f"  Number of points in LAS file: {len(las.points)}")

    # XYZ coordinates
    if xyz:
        pos = torch.stack([
            torch.from_numpy(np.array(las[axis])) for axis in ["X", "Y", "Z"]
        ], dim=-1)
        pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    # RGB colors
    if rgb:
        data.rgb = to_float_rgb(torch.stack([
            torch.from_numpy(np.array(las[axis], dtype=np.float32) / 65535) 
            for axis in ["red", "green", "blue"]
        ], dim=-1))

    # Intensity
    if intensity:
        data.intensity = torch.from_numpy(
            np.array(las['intensity'], dtype=np.float32)
        ).clip(min=0, max=max_intensity) / max_intensity

    # Semantic labels
    if semantic:
        y = torch.LongTensor(las['classification'])
        if remap:
            max_class = int(y.max())
            if max_class >= len(ID2TRAINID):
                extended = np.full(max_class + 1, fill_value=VANCOUVER_NUM_CLASSES, dtype=np.int64)
                extended[:len(ID2TRAINID)] = ID2TRAINID
                mapping = extended
            else:
                mapping = ID2TRAINID
            data.y = torch.from_numpy(mapping)[y]
        else:
            data.y = y

    # Instance labels not supported
    if instance:
        raise NotImplementedError("The dataset does not contain instance labels.")

    return data


def change_classifications(input_file, output_file, new_classifications):
    """
    Change classifications in a LAS file and save to new file.
    
    Args:
        input_file: Path to input .las file
        output_file: Path to save modified .las file
        new_classifications: New classification array
    """
    las = laspy.read(input_file)
    las.classification = new_classifications
    
    # Assign CRS
    las.header.crs = CRS.from_epsg(7416)
    las.header.global_encoding.wkt = True
    
    las.write(output_file)
    print(f"  Saved predictions to: {output_file}")


def load_dataset_config(config_str):
    """Load dataset-specific configuration based on config string."""
    global VANCOUVER_CLASS_COLORS, VANCOUVER_CLASS_NAMES, ID2TRAINID, VANCOUVER_NUM_CLASSES
    
    # Check if using KDS configuration
    if 'kds' in config_str.lower():
        try:
            import src.datasets.kds_config as kds_cfg
            VANCOUVER_CLASS_COLORS = kds_cfg.CLASS_COLORS
            VANCOUVER_CLASS_NAMES = kds_cfg.CLASS_NAMES
            ID2TRAINID = kds_cfg.ID2TRAINID
            VANCOUVER_NUM_CLASSES = kds_cfg.KDS_NUM_CLASSES
            print(f"  Loaded KDS dataset configuration")
        except ImportError:
            print(f"  Warning: Could not import KDS config, using Vancouver config")
    
    print(f"  Number of classes: {len(VANCOUVER_CLASS_COLORS)}")


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def process_file(input_file, output_folder, cfg, transforms_dict, model):
    """
    Process a single LAS/LAZ file.
    
    Args:
        input_file: Path to input file
        output_folder: Path to output directory
        cfg: Configuration object
        transforms_dict: Dictionary of transforms
        model: Pretrained model
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")
    
    # 1. Read input data
    print("  Step 1: Reading input data")
    data = read_vancouver_tile(str(input_file))
    
    # 2. Apply pre-transforms
    print("  Step 2: Applying pre-transforms")
    nag = transforms_dict['pre_transform'](data)
    
    # 3. Simulate dataset I/O behavior
    print("  Step 3: Simulating dataset I/O")
    nag = NAGRemoveKeys(
        level=0, 
        keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys]
    )(nag)
    nag = NAGRemoveKeys(
        level='1+', 
        keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys]
    )(nag)
    
    # 4. Move to device and apply on-device transforms
    print("  Step 4: Moving to GPU and applying on-device transforms")
    nag = nag.cuda()
    nag = transforms_dict['on_device_test_transform'](nag)
    
    # 5. Run inference
    print("  Step 5: Running inference")
    with torch.no_grad():
        output = model(nag)
    
    # 6. Get full-resolution predictions
    print("  Step 6: Computing full-resolution predictions")
    nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
    
    raw_semseg_y = output.full_res_semantic_pred(
        super_index_level0_to_level1=nag[0].super_index,
        sub_level0_to_raw=nag[0].sub
    )
    
    # 7. Save results
    print("  Step 7: Saving results")
    output_file = output_folder / input_file.name
    change_classifications(
        input_file=str(input_file),
        output_file=str(output_file),
        new_classifications=raw_semseg_y.cpu()
    )
    
    print(f"  ✓ Successfully processed {input_file.name}")


def main():
    """Main inference pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Semantic segmentation inference on LAS/LAZ files using Superpoint Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python predict.py --inputlaz data/input.las --output_folder results/
  
  # Process a directory of files
  python predict.py --inputlaz data/tiles/ --output_folder results/
  
  # Use custom checkpoint and config
  python predict.py --inputlaz data/ --output_folder results/ \\
      --ckpt_path models/custom.ckpt \\
      --config experiment=semantic/custom
        """
    )
    
    parser.add_argument(
        '--inputlaz',
        type=str,
        required=True,
        help='Path to input LAS/LAZ file or directory containing LAS/LAZ files'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to output folder (will be created if it does not exist)'
    )
    
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='/home/rajoh/projects/superpoint_transformer/logs/kdsvox025/runs/2025-09-29_17-16-08/checkpoints/last.ckpt',
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='experiment=semantic/vox025kds',
        help='Configuration string for the experiment'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("SEMANTIC SEGMENTATION INFERENCE")
    print("="*70)
    print(f"Input: {args.inputlaz}")
    print(f"Output folder: {args.output_folder}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Config: {args.config}")
    print("="*70)
    
    # Get input files
    try:
        input_files = get_input_files(args.inputlaz)
        print(f"\nFound {len(input_files)} file(s) to process:")
        for f in input_files:
            print(f"  - {f.name}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder ready: {output_folder}")
    
    # Check checkpoint exists
    if not Path(args.ckpt_path).exists():
        print(f"Error: Checkpoint file not found: {args.ckpt_path}")
        sys.exit(1)
    
    # Load dataset configuration
    print(f"\n{'='*70}")
    print("Loading dataset configuration")
    print(f"{'='*70}")
    load_dataset_config(args.config)
    
    # Parse configuration
    print(f"\n{'='*70}")
    print("Parsing model configuration")
    print(f"{'='*70}")
    cfg = init_config(overrides=[args.config, f"datamodule.load_full_res_idx={True}"])
    print("  Configuration loaded successfully")
    
    # Instantiate transforms
    print(f"\n{'='*70}")
    print("Instantiating transforms")
    print(f"{'='*70}")
    from src.transforms import instantiate_datamodule_transforms
    transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)
    print("  Transforms instantiated")
    
    # Load model
    print(f"\n{'='*70}")
    print("Loading pretrained model")
    print(f"{'='*70}")
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(args.ckpt_path)
    model = model.eval().cuda()
    print("  Model loaded and set to evaluation mode")
    
    # Process all files
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(input_files)} FILE(S)")
    print(f"{'='*70}")
    
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(input_files, 1):
        try:
            print(f"\n[{i}/{len(input_files)}]", end=" ")
            process_file(input_file, output_folder, cfg, transforms_dict, model)
            successful += 1
        except Exception as e:
            print(f"  ✗ Error processing {input_file.name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {successful}/{len(input_files)} files")
    if failed > 0:
        print(f"Failed: {failed}/{len(input_files)} files")
    print(f"Results saved to: {output_folder}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
