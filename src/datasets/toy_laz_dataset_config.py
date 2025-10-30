import numpy as np


########################################################################
#                         Download information                         #
########################################################################

FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977'

# DALES in LAS format
LAS_TAR_NAME = 'dales_semantic_segmentation_las.tar.gz'
LAS_UNTAR_NAME = "dales_las"

# DALES in PLY format
PLY_TAR_NAME = 'dales_semantic_segmentation_ply.tar.gz'
PLY_UNTAR_NAME = "dales_ply"

# DALES in PLY, only version with intensity and instance labels
OBJECTS_TAR_NAME = 'DALESObjects.tar.gz'
OBJECTS_UNTAR_NAME = "DALESObjects"


########################################################################
#                              Data splits                             #
########################################################################

TILES = {
    'train': [
        '1km_6143_589'],
    'val': [
        '1km_6143_590'],

    'test': [
        '1km_6147_588']}





########################################################################
#                                Labels                                #
########################################################################



TOY_DATASET_NUM_CLASSES = 10

#ID2TRAINID = np.asarray([0,2,3,4,5,6,7,9,14,17,18])

import numpy as np

ID2TRAINID = 0 * np.ones(19, dtype=int)  # 0..18  (im asuming that 0 and 1 shoudl be training id 0 == ignore )
ID2TRAINID[2] = 0    # Ground
ID2TRAINID[3] = 1    # Low vegetation
ID2TRAINID[4] = 2    # Medium vegetation
ID2TRAINID[5] = 3    # High vegetation
ID2TRAINID[6] = 4    # Building
ID2TRAINID[7] = 5    # Low point (noise)
ID2TRAINID[9] = 6    # Water
ID2TRAINID[14] = 7   # Wire
ID2TRAINID[17] = 8   # Bridge deck
ID2TRAINID[18] = 9   # Highnoise


CLASS_NAMES = [
    'Ground',
    'LowVegetation',
    'MediumVegetation',
    'HighVegetation',
    'Building',
    'LowpointNoice',
    'Water',
    'wire',
    'BridgeDeck',
    'HighNoise',
    'Something']




CLASS_COLORS = np.asarray([
    [10, 10, 10],  #  unknown color
    [243, 214, 171],  # sunset
    [ 70, 115,  66],  # fern green
    [233,  50, 239],
    [243, 238,   0],
    [190, 153, 153],
    [  0, 233,  11],
    [0, 0,   256],     # lets make water blue
    [239, 114,   0],
    [214,   66,  54],  # vermillon
    [  0,   8, 116]])

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [2, 3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(TOY_DATASET_NUM_CLASSES) if not i in THING_CLASSES]
