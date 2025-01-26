#!/usr/bin/env python
"""
This script uses the camera and sonar metadata to build a dataset.
"""

# Import external libraries 
from typing import List, Optional
import pandas as pd 
import numpy as np
import shutil 
import cv2 
import os 

# Import internal scripts
from src.bag_reader import BagReader, BagReaderException
from src.utils import sonar_utils, camera_utils
from src import config 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

class BuilderException(Exception):
    """Custom exception for Builder class 

    Args:
        Exception (_type_): inherits from base Exception class 
    """
    def __init__(self, message):
        """Constructor for the BuilderException class. 
        It prints out an exception message 

        Args:
            message (str): Exception message
        """
        super().__init__(f"BuilderException: {message}")

class Builder:
    """This class saves images to disk and moves images to different locations on the disk."""

    def __init__(self, metadata_df: pd.DataFrame, 
                       image_locations_on_disk: Optional[List[str]] = None):
        """Constructor for the Builder class. 

        Args:
            metadata_df (pd.DataFrame): Metadata 
            image_locations_on_disk (List[str] | None, optional): list of image locations already present on disk. Defaults to None.

        Raises:
            BuilderException: metadata is empty 
        """
        
        if metadata_df is None:
            raise BuilderException("No inputs selected. Nothing to build!") 
        
        if image_locations_on_disk is not None and metadata_df.shape[0] == len(image_locations_on_disk):
            self.image_locations_on_disk = np.asarray(image_locations_on_disk) if not isinstance(image_locations_on_disk, np.ndarray) else image_locations_on_disk
        else:
            self.image_locations_on_disk = np.empty(metadata_df.shape[0])
            self.image_locations_on_disk[:] = np.nan
        
        
    def copy(self, source_image_file_path: str, 
                   destination_image_file_path: str) -> None:
        """Copies an image to another location

        Args:
            source_image_file_path (str): Source file path
            destination_image_file_path (str): Destination file path 

        Raises:
            BuilderException: Source image does not exist 
        """

        if not os.path.exists(source_image_file_path):
            raise BuilderException("Image does not exists!")
        
        shutil.copy(source_image_file_path, destination_image_file_path)
        
    def save(self, image: np.ndarray, 
                   image_file_path: str) -> None:
        """Saves image on disk

        Args:
            image (np.ndarray): Image
            image_file_path (str): Destination file path

        Raises:
            BuilderException: image already exists 
        """

        if not os.path.exists(image_file_path):
            cv2.imwrite(image_file_path, image)
        else:
            raise BuilderException("Image already exists!")

    def progress_bar(self, count_value: int, 
                           total: int) -> None:
        """Progress bar 

        Args:
            count_value (int): progress value of bar
            total (int): max value of progress bar 
        """

        bar_length = 75
        filled_up_Length = int(round(bar_length* count_value / float(total)))
        percentage = round(100.0 * count_value/float(total),1)
        bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        print('[%s] %s%s ...\r' %(bar, percentage, '%'), end='\r', flush=True)


class XTFPatchBuilder(Builder):
    """This class saves sonar images to disk. 

    Args:
        Builder (_type_): inherits from base Builder class 
    """

    def __init__(self, metadata_df: pd.DataFrame, 
                       image_locations_on_disk: Optional[List[str]] = None):
        """Constructor for the XTFPatchBuilder class. 
        Saves sonar images on disk using the metadata except for the images that are already present on disk, 
        it copies those from their current location to their destination location found in the metadata.

        Args:
            metadata_df (pd.DataFrame): metadata
            image_locations_on_disk (List[str] | None, optional): list of image locations already present on disk. Defaults to None.
        """

        super().__init__(metadata_df, image_locations_on_disk)

        print("Building acoustic dataset...")

        counter = 1
        for xtf_file_name in np.unique(metadata_df["XTF_File_Name"]):

            # Get all patches to be extracted from current XTF 
            row_idxs = np.where((metadata_df["XTF_File_Name"] == xtf_file_name).to_numpy())[0]
            original_image_paths = self.image_locations_on_disk[row_idxs]

            # Extract XTF pings
            if np.any(np.isnan(original_image_paths)):
                xtf_file_path = os.path.join(config.XTF_FOLDER_PATH, xtf_file_name)
                xtf_pings = sonar_utils.load_xtf(xtf_file_path)
                waterfall = sonar_utils.calculate_waterfall(xtf_pings)

            # Extract patches and save them to disk
            for i, row_idx in enumerate(row_idxs):
                patch_file_path = metadata_df["Patch_File_Path"].iloc[row_idx]

                # Create folder if it does not exist 
                patch_folder_path = os.path.dirname(patch_file_path)
                if not os.path.exists(patch_folder_path):
                    os.makedirs(patch_folder_path)

                # If already present copy it else extract it and save it 
                if not np.isnan(original_image_paths[i]):
                    try:
                        self.copy(original_image_paths[i], patch_file_path)
                    except BuilderException as e:
                        print(f"{e} Skipping patch {original_image_paths[i]} inside {xtf_file_name}...")
                else:
                    try:
                        patch = waterfall[metadata_df["Start_Ping_Index"].iloc[row_idx]:metadata_df["End_Ping_Index"].iloc[row_idx],
                                          metadata_df["Start_Bin_Index"].iloc[row_idx]:metadata_df["End_Bin_Index"].iloc[row_idx]]
                        self.save(patch, patch_file_path)
                    except BuilderException as e:
                        print(f"{e} Skipping patch inside {xtf_file_name} between pings ({metadata_df['Start_Ping_Index'].iloc[row_idx]}, \
                            {metadata_df['End_Ping_Index'].iloc[row_idx]}) and bins ({metadata_df['Start_Bin_Index'].iloc[row_idx]}, \
                            {metadata_df['End_Bin_Index'].iloc[row_idx]})...")

                self.progress_bar(counter, metadata_df.shape[0])
                counter += 1

        print("Acoustic dataset built!")

class CameraImageBuilder(Builder):
    """This class saves camera images to disk. 

    Args:
        Builder (_type_): inherits from base Builder class 
    """

    def __init__(self, metadata_df: pd.DataFrame, 
                       image_locations_on_disk: Optional[List[str]] = None):
        """Constructor for the CameraImageBuilder class. 
        Saves camera images on disk using the metadata except for the images that are already present on disk, 
        it copies those from their current location to their destination location found in the metadata.

        Args:
            metadata_df (pd.DataFrame): metadata
            image_locations_on_disk (List[str] | None, optional): list of image locations already present on disk. Defaults to None.
        """
        
        super().__init__(metadata_df, image_locations_on_disk)

        print("Building optical dataset...")

        self.counter = 1
        for bag_file_name in np.unique(metadata_df["ROS_Bag_File_Name"]):

            # Get all images to be extracted from current ROS Bag 
            row_idxs = np.where((metadata_df["ROS_Bag_File_Name"] == bag_file_name).to_numpy())[0]
            original_image_paths = self.image_locations_on_disk[row_idxs]

            # If all images are already on disk
            if not np.any(np.isnan(original_image_paths)):
                self.copy_images_from_disk(row_idxs, metadata_df, original_image_paths, bag_file_name)
                continue 

            # Create ROS Bag reader and load ROS Bag 
            bag_file_path = os.path.join(config.ROSBAG_FOLDER_PATH, bag_file_name)
            try:
                reader = BagReader(bag_file_path, config.USING_ROS2)
                reader.load_bag()
            except BagReaderException as e:
                print(f"{e} Skipping...")
                continue

            topics = np.unique(metadata_df["Topic_Name"])
            sequence_indices = {topic: 0 for topic in topics}

            # Iterate over ROS Bag messages 
            message_gen = reader.extract_messages(topics)
            while True:
                try:
                    # Extract message and corresponding topic name 
                    topic, msg = next(message_gen)

                    # Get row number
                    row_idx = np.where((metadata_df["ROS_Bag_File_Name"] == bag_file_name) &
                                       (metadata_df["Topic_Name"] == topic) & 
                                       (metadata_df["Sequence_Index"] == sequence_indices[topic]))[0]
                    
                    if len(row_idx) == 0:
                        sequence_indices[topic] += 1  
                        continue

                    # Create folder if it does not exist 
                    image_file_path = metadata_df["Image_File_Path"].iloc[row_idx[0]]

                    image_folder_path = os.path.dirname(image_file_path)
                    if not os.path.exists(image_folder_path):
                        os.makedirs(image_folder_path)

                    # If already present copy it else extract it and save it 
                    original_image_idx = np.where(row_idxs == row_idx[0])[0][0]
                    if not np.isnan(original_image_paths[original_image_idx]):
                        try:
                            self.copy(original_image_paths[original_image_idx], image_file_path)
                        except BuilderException as e:
                            print(f"{e} Skipping image {original_image_paths[original_image_idx]} inside {bag_file_name}...")
                    else:
                        try:
                            image = camera_utils.extract_image(msg)
                            self.save(image, image_file_path)
                        except BuilderException as e:
                            print(f"{e} Skipping image inside {bag_file_name} in topic {topic} at sequence index {sequence_indices[topic]}...")

                    self.progress_bar(self.counter, metadata_df.shape[0])
                    self.counter += 1

                    sequence_indices[topic] += 1  
                except StopIteration:
                    break
        print("Optical dataset built!")


    def copy_images_from_disk(self, image_row_idxs: np.ndarray, 
                                    metadata_df: pd.DataFrame, 
                                    original_image_paths: List[str], 
                                    bag_file_name: str) -> None:
        """Copies images that are already present on disk to destination file path inside metadata  

        Args:
            image_row_idxs (np.ndarray): indices of images that are already present on disk 
            metadata_df (pd.DataFrame): metadata
            original_image_paths (List[str]): list of file paths of images that are already present on disk 
            bag_file_name (str): ROS bag file name
        """

        for i, row_idx in enumerate(image_row_idxs):

            # Create folder if it does not exist 
            image_file_path = metadata_df["Image_File_Path"].iloc[row_idx[0]]
            image_folder_path = os.dirname(image_file_path)
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)

            # Copy image and save it to disk 
            try:
                self.copy(original_image_paths[i], image_file_path)
            except BuilderException as e:
                print(f"{e} Skipping image {original_image_paths[i]} inside {bag_file_name}...")

            self.progress_bar(self.counter, metadata_df.shape[0])
            self.counter += 1
    