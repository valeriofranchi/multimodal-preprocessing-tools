#!/usr/bin/env python
"""
Configuration file
"""

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

# GLOBAL PARAMETERS

############ Patch parameters ##############

PATCH_HEIGHT = 256
PATCH_WIDTH = 256
STRIDE_X = 128
STRIDE_Y = 128

############ XTF parameters ##############

EPSG_CODE = 25831 
SONAR_RPY_RAD = False
XTF_UTC_TIME = 0

############ Image filtering parameters ##############

MAX_IMAGE_HEIGHT = 5.0
PERCENT_IMAGE_OVERLAP = 0.4
SKIP_FRAMES = 2

############ File locations ##############

XTF_FOLDER_PATH = "raw/sss"
ROSBAG_FOLDER_PATH = "raw/bags"
PROCESS_FOLDER_PATH = "/media/vicorob/Filesystem2/test10/processed"
DATASET_FOLDER_PATH = "/media/vicorob/Filesystem2/test10/dataset"

############ ROS Parameters ##############

NAVIGATION_TOPIC = "/girona1000/navigator/navigation_throttle"
CAMERA_TOPICS = ["/girona1000/left/flir_spinnaker_camera/image_raw/compressed",
                 "/girona1000/right/flir_spinnaker_camera/image_raw/compressed"]
CAMERA_INFO_TOPICS = ["/girona1000/left/flir_spinnaker_camera/image_raw/camera_info",
                      "/girona1000/right/flir_spinnaker_camera/image_raw/camera_info"]
USING_ROS2 = False
ROSBAG_UTC_TIME = 2

RUN_ALL = False
EXTRACT_PATCHES = False
EXTRACT_NAVIGATION = False
EXTRACT_IMAGES = False
MATCH_IMAGES_TO_NAVIGATION = False
MATCH_IMAGES_TO_PATCHES = True

DEBUG_MODE = True
WAIT_FOR_USER_INPUT = False

SAVE_PATCHES = False
SAVE_IMAGES = False
BUILD_DATASET = False

def printParameters():
    """Prints parameters present in the configuration file
    """

    print("LISTING PARAMETERS...\n")
    print("Patch parameters:")
    print(f"\tPatch shape: ({PATCH_HEIGHT}, {PATCH_WIDTH})")
    print(f"\tStride: ({STRIDE_X}, {STRIDE_Y})")
    print(f"EPSG Code: {EPSG_CODE}")
    print(f"ROS Version: {'2' if USING_ROS2 else '1'}")
    print(f"ROS Navigation Topic: {NAVIGATION_TOPIC}")
    print(f"ROS Image Topics")
    for i,topic in enumerate(CAMERA_TOPICS):
        print(f"\t{topic} ({i+1}/{len(CAMERA_TOPICS)})")
    print(f"ROS Camera Info Topics")
    for i,topic in enumerate(CAMERA_INFO_TOPICS):
        print(f"\t{topic} ({i+1}/{len(CAMERA_INFO_TOPICS)})")
    print(f"Maximum vehicle height for camera images: {MAX_IMAGE_HEIGHT} [metres]")
    print("ROS Bag parameters:")
    print(f"\tTime: UTC{'+' if ROSBAG_UTC_TIME >= 0 else '-'}{abs(ROSBAG_UTC_TIME)}")
    print("XTF parameters:")
    print(f"\tRoll, pitch, yaw in radians: {'Yes' if SONAR_RPY_RAD else 'No'}")
    print(f"\tTime: UTC{'+' if XTF_UTC_TIME >= 0 else '-'}{abs(XTF_UTC_TIME)}")
    print(f"Path parameters:")
    print(f"\tXTF folder: {XTF_FOLDER_PATH}")
    print(f"\tROS Bag folder: {ROSBAG_FOLDER_PATH}")
    print(f"\tProcessed data folder: {PROCESS_FOLDER_PATH}")
    print(f"\tDataset Folder path: {DATASET_FOLDER_PATH}")
    print("Additional parameters:")
    print(f"\tRunning in DEBUG Mode: {'Yes' if DEBUG_MODE else 'No'}")
    print(f"\tWait for user input to continue: {'Yes' if WAIT_FOR_USER_INPUT else 'No'}")
    print(f"\tSave XTF patches: {'Yes' if SAVE_PATCHES else 'No'}")
    print(f"\tSave camera images: {'Yes' if SAVE_IMAGES else 'No'}")


def saveParameters():
    """Saves certain parameters from the configuration file to a json file
    """

    params = { "patch_info": {
                    "patch_height": PATCH_HEIGHT,
                    "patch_width": PATCH_WIDTH,
                    "stride_x": STRIDE_X,
                    "stride_y": STRIDE_Y,
            }, "xtf_info": {
                    "epsg_code": EPSG_CODE, 
                    "utc_time": XTF_UTC_TIME
            }, "image_preprocess": {
                    "max_altitude": MAX_IMAGE_HEIGHT,
                    "percent_image_overlap": PERCENT_IMAGE_OVERLAP,
                    "skipped_frames": SKIP_FRAMES
            }, "ros_info": {
                "ros2": USING_ROS2,
                "utc_time": ROSBAG_UTC_TIME
            }
    }

    import os 
    import json
    def convert(x):
        if hasattr(x, "tolist"):  
            return x.tolist()
        raise TypeError(x)

    # save data to preprocessing folder 
    if not os.path.exists(PROCESS_FOLDER_PATH):
        os.makedirs(PROCESS_FOLDER_PATH)
    json_file_path = os.path.join(PROCESS_FOLDER_PATH, "params.json")
    with open(json_file_path, 'w') as f:
        json.dump(params, f, default=convert, indent=4) 

    # save data to dataset folder 
    if not os.path.exists(DATASET_FOLDER_PATH):
        os.makedirs(DATASET_FOLDER_PATH)
    json_file_path = os.path.join(DATASET_FOLDER_PATH, "params.json")
    with open(json_file_path, 'w') as f:
        json.dump(params, f, default=convert, indent=4) 

    print("Parameters saved to disk!")  
    