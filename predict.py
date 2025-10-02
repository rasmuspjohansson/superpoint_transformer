#!/usr/bin/env python
"""
Semantic Segmentation Prediction Script
Converts Jupyter notebook predict.ipynb to standalone Python script
"""

import os
import sys
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
# Configuration Settings
# ============================================================================

# Choose voxel setup: 1 = 0.1, 2 = 0.175, 3 = 0.25, 4-6 = KDS variants
CHOICE = 6

voxel_settings = {
    1: {
        "ckpt_path": "/mnt/T/mnt/logs_and_models/dales_trained_spt/spt-2_dales.ckpt",
        "config": "experiment=semantic/dales",
        "output_las_name": "vox01"
    },
    2: {
        "ckpt_path": "/home/rajoh/projects/superpoint_transformer_rasmus_version/logs/train/runs/2025-09-18_14-56-24/checkpoints/last.ckpt",
        "config": "experiment=semantic/adjusteddales",
        "output_las_name": "vox_o175"
    },
    3: {
        "ckpt_path": "/home/rajoh/projects/superpoint_transformer_rasmus_version/logs/train/runs/2025-09-19_13-15-16/checkpoints/last.ckpt",
        "config": "experiment=semantic/vox025dales",
        "output_las_name": "vox_o25"
    },
    4: {
        "ckpt_path": "/home/rajoh/projects/superpoint_transformer/logs/train/runs/2025-09-23_12-54-10/checkpoints/last.ckpt",
        "config": "experiment=semantic/vox01kdsdata",
        "output_las_name": "vox_kds_data_01"
    },
    5: {
        "ckpt_path": "/home/rajoh/projects/superpoint_transformer/logs/kdsvox025/runs/2025-09-23_14-50-17/checkpoints/last.ckpt",
        "config": "experiment=semantic/vox025kds",
        "output_las_name": "vox025kds"
    },
    6: {
        "ckpt_path": "/home/rajoh/projects/superpoint_transformer/logs/kdsvox025/runs/2025-09-29_17-16-08/checkpoints/last.ckpt",
        "config": "experiment=semantic/vox025kds",
        "output_las_name": "vox025kds_larger_trainingset"
    }
}

# Input file path
FILEPATH = "/home/rajoh/projects/superpoint_transformer/data/kds/raw/train/1km_6139_588.laz"

# Apply settings
settings = voxel_settings[CHOICE]
ckpt_path = settings["ckpt_path"]
config = settings["config"]
output_las_name = settings["output_las_name"]

print("=" * 70)
print("Using settings:")
print(f"  Checkpoint: {ckpt_path}")
print(f"  Config: {config}")
print(f"  Input LAS: {FILEPATH}")
print(f"  Output LAS name: {output_las_name}")
print("=" * 70)


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

# Override with KDS config if using KDS models (choice > 3)
if CHOICE > 3:
    from src.datasets.kds_config import *
    VANCOUVER_CLASS_COLORS = CLASS_COLORS
    VANCOUVER_CLASS_NAMES = CLASS_NAMES
    ID2TRAINID = ID2TRAINID
    VANCOUVER_NUM_CLASSES = KDS_NUM_CLASSES

print(f"Number of classes: {len(VANCOUVER_CLASS_COLORS)}")


# ============================================================================
# Helper Functions
# ============================================================================

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

    print(f"Number of points in LAS file: {len(las.points)}")

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
    
    print(f"Global encoding: {las.header.global_encoding}")
    las.write(output_file)
    print(f"Saved predictions to: {output_file}")


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def main():
    """Main inference pipeline."""
    
    # 1. Read input data
    print("\n" + "=" * 70)
    print("Step 1: Reading input data")
    print("=" * 70)
    data = read_vancouver_tile(FILEPATH)
    input_las = laspy.read(FILEPATH)
    header = input_las.header
    print(f"Data loaded. Number of points: {data.num_points}")
    
    # 2. Parse configuration
    print("\n" + "=" * 70)
    print("Step 2: Parsing configuration")
    print("=" * 70)
    cfg = init_config(overrides=[config, f"datamodule.load_full_res_idx={True}"])
    print("Configuration loaded successfully")
    
    # 3. Instantiate transforms
    print("\n" + "=" * 70)
    print("Step 3: Instantiating transforms")
    print("=" * 70)
    from src.transforms import instantiate_datamodule_transforms
    transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)
    print("Transforms instantiated")
    
    # 4. Apply pre-transforms
    print("\n" + "=" * 70)
    print("Step 4: Applying pre-transforms")
    print("=" * 70)
    nag = transforms_dict['pre_transform'](data)
    print(f"Pre-transforms applied. NAG structure: {nag}")
    
    # 5. Simulate dataset I/O behavior
    print("\n" + "=" * 70)
    print("Step 5: Simulating dataset I/O")
    print("=" * 70)
    nag = NAGRemoveKeys(
        level=0, 
        keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys]
    )(nag)
    nag = NAGRemoveKeys(
        level='1+', 
        keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys]
    )(nag)
    print("Keys removed to match training I/O")
    
    # 6. Move to device and apply on-device transforms
    print("\n" + "=" * 70)
    print("Step 6: Moving to GPU and applying on-device transforms")
    print("=" * 70)
    nag = nag.cuda()
    nag = transforms_dict['on_device_test_transform'](nag)
    print("On-device transforms applied")
    
    # 7. Load model
    print("\n" + "=" * 70)
    print("Step 7: Loading pretrained model")
    print("=" * 70)
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(ckpt_path)
    model = model.eval().to(nag.device)
    print("Model loaded and set to evaluation mode")
    
    # 8. Run inference
    print("\n" + "=" * 70)
    print("Step 8: Running inference")
    print("=" * 70)
    with torch.no_grad():
        output = model(nag)
    print(f"Inference complete. Predictions shape: {output.semantic_pred().shape}")
    
    # 9. Get full-resolution predictions
    print("\n" + "=" * 70)
    print("Step 9: Computing full-resolution predictions")
    print("=" * 70)
    nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
    
    raw_semseg_y = output.full_res_semantic_pred(
        super_index_level0_to_level1=nag[0].super_index,
        sub_level0_to_raw=nag[0].sub
    )
    print(f"Full-resolution predictions computed. Shape: {raw_semseg_y.shape}")
    
    # 10. Save results
    print("\n" + "=" * 70)
    print("Step 10: Saving results")
    print("=" * 70)
    output_file = output_las_name + "_all_points.las"
    change_classifications(
        input_file=FILEPATH,
        output_file=output_file,
        new_classifications=raw_semseg_y.cpu()
    )
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
