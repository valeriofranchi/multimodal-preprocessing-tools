#!/usr/bin/env python
"""
This script performs camera image extraction from ROS bags. 
It filters the images based on configuration parameters, 
and it plots certain extracted images for debugging purposes.
"""

# Import external libraries 
from typing import List, Tuple, Dict, Union 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import json 
import cv2
import os 

# Import internal scripts
from src.builder import CameraImageBuilder, BuilderException 
from src.bag_reader import BagReader, BagReaderException
from src.utils import camera_utils, common_utils
from src import config 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

class CameraExtractor:
    """This class extracts images from several ROS bags and stores metadata information of every image. """

    def __init__(self):
        """Constructor for the CameraExtractor class. 
        It processes every ROS Bag, extracts the images and stores the metadata."""

        print("\nEXTRACTING CAMERA IMAGES...")
        
        # Create subfolder
        self.camera_folder = os.path.join(config.PROCESS_FOLDER_PATH, "camera")
        if not os.path.exists(self.camera_folder):
            os.makedirs(self.camera_folder)

        # Define metadata file header
        header = ["ROS_Bag_File_Name","Sequence_Index","Camera_Index","Topic_Name","Timestamp_Seconds","Image_File_Path"]

        # Load calibration data 
        with open(os.path.join(config.ROSBAG_FOLDER_PATH, "calibration.json")) as json_file:
            self.calib_data = json.load(json_file)

        # Process ROS Bags 
        bag_file_names = os.listdir(config.ROSBAG_FOLDER_PATH)
        for i, bag_file_name in enumerate(bag_file_names):
            
            if config.WAIT_FOR_USER_INPUT:
                input("\nPress Enter to continue...")

            bag_file_path = os.path.join(config.ROSBAG_FOLDER_PATH, bag_file_name)
            print(f"\nProcessing {i+1}/{len(bag_file_names)}:  {bag_file_path}...")

            # Check if camera metadata file is already present on the disk 
            self.bag_folder = os.path.join(self.camera_folder, bag_file_name.split(".bag")[0])
            camera_csv_file_path = os.path.join(self.bag_folder, "camera.csv") 
            if os.path.exists(self.bag_folder) and os.path.exists(camera_csv_file_path):
                print(f"Folder {self.bag_folder} already present!")
                continue

            # Create ROS Bag reader and load ROS Bag 
            try:
                reader = BagReader(bag_file_path, config.USING_ROS2)
                reader.load_bag()
            except BagReaderException as be:
                print(f"{be} Skipping...")
                continue
            except TypeError as te:
                print(f"{te} Skipping...")
                continue

            # Check which camera topics are in ROS Bag 
            camera_topics = [topic for topic in config.CAMERA_TOPICS if reader.check_if_topic_in_bag(topic)]
            if len(camera_topics) == 0:
                print("Invalid camera topic(s). Skipping bag...")
                continue

            # Process rosbag 
            topic_metadata, topic_info = self.process_bag(reader, camera_topics)
            print("Bag processed!")

            # Create bag folder if it does not exist 
            if not os.path.exists(self.bag_folder):
                os.makedirs(self.bag_folder)
            camera_df = pd.DataFrame(topic_metadata, columns=header)

            if len(topic_metadata) > 0:
                if config.SAVE_IMAGES:
                    try:
                        CameraImageBuilder(camera_df)
                    except BuilderException as e:
                        print(f"{e} Skipping...")
            else:
                print("No information!")

            # Save data to disk 
            camera_df.to_csv(camera_csv_file_path, index=False)
            json_file_path = os.path.join(self.bag_folder, "info.json")
            with open(json_file_path, 'w') as f:
                json.dump(topic_info, f, default=common_utils.convert, indent=4) 
            if len(topic_metadata) > 0:
                print("Metadata saved to disk!")  
               
                

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PREPROCESSING FUNCTIONS                                                                       #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 

    
    def process_bag(self, reader: BagReader, 
                          camera_topics: List[str]) -> Tuple[List[List[Union[str, float, int]]], Dict[str, Union[str, int, List[float], List[List[float]]]]]:
        """Processes the rosbag and extracts metadata information from the ROS image messages.

        Args:
            reader (BagReader): BagReader object
            camera_topics (List[str]): List of camera image topics 

        Returns:
            Tuple[List[List[str | float | int]], Dict[str, str | int | List[float] | List[List[float]]]]: Metadata per topic, and calibration data per topic
        """

        if config.DEBUG_MODE:    
            total_images_selected = 0
            max_sample_images_per_topic = 5
            sample_images_per_topic = {topic: [] for topic in camera_topics}

        image_sequence_indices = {topic: -1 for topic in camera_topics}
        skipped = {topic: 0 for topic in camera_topics}

        # Retrieve number of image messages
        num_messages = 0
        for topic in camera_topics:
            try:
                num_messages += reader.message_count(topic)
            except BagReaderException as e:
                print(e)

        # Define image folder names
        topic_folder_paths = {topic: os.path.join(self.bag_folder, "camera_" + str(i).zfill(len(str(len(camera_topics))))) for i, topic in enumerate(camera_topics)}

        # Define calibration data 
        topic_info = {}
        for i, topic in enumerate(camera_topics):
            try:
                topic_calib = self.calib_data[reader.get_bag_file_name().split(".bag")[0]]["left" if "left" in topic else "right" if "right" in topic else None]
                topic_info["camera_" + str(i).zfill(len(str(len(camera_topics))))] = {"topic": topic, "height": topic_calib["height"], "width": topic_calib["width"], "intrinsics": np.array(topic_calib["K"]), 
                                                                                      "distortion": np.array(topic_calib["D"]), "extrinsics": np.array(topic_calib["T"])}
            except KeyError:
                pass
        
        # Iterate over ROS Bag messages 
        topic_metadata = []
        message_gen = reader.extract_messages(camera_topics)
        while True:
            try:
                # Extract message, corresponding topic name and timestamp 
                topic, msg = next(message_gen)
                timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9  

                if topic in camera_topics:
                    image_sequence_indices[topic] += 1  

                # Process current image 
                if topic in camera_topics:

                        # Skip frame
                        if skipped[topic] < config.SKIP_FRAMES:
                            skipped[topic] += 1
                            continue 

                        # Define image file path
                        image_file_name = str(image_sequence_indices[topic]).zfill(len(str(num_messages))) + ".png" 
                        image_file_path = os.path.join(topic_folder_paths[topic], image_file_name)
                        
                        # Define metadata for current image 
                        camera_index = int(topic_folder_paths[topic].split("/")[-1].split("_")[-1])
                        topic_metadata.append([reader.get_bag_file_name(), image_sequence_indices[topic], camera_index, topic, timestamp, image_file_path])
                         
                        if config.DEBUG_MODE:
                            total_images_selected += 1
                            if len(sample_images_per_topic[topic]) < max_sample_images_per_topic:
                                sample_images_per_topic[topic].append(camera_utils.extract_image(msg))
                    
                        skipped[topic] = 0  

            except StopIteration:
                break

        if config.DEBUG_MODE:
            print(f"Number of images inside bag: {num_messages}")
            print(f"Number of images after filtering: {total_images_selected}") 
            self.plot_images(sample_images_per_topic)

        return topic_metadata, topic_info
    

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PLOTTING FUNCTIONS                                                                            #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 


    def plot_images(self, images_per_topic: Dict[str, List[np.ndarray]]) -> None:
        """Plots images belonging to different topics on a matplotlib figure. 

        Args:
            images_per_topic (Dict[str, List[np.ndarray]]): Dictionary containing several images per camera topic 
        """

        # Calculate number of columns needed 
        num_columns = np.max([len(images) for _, images in images_per_topic.items()]) 
        if num_columns > 0:

            # Plot images     
            fig, ax = plt.subplots(len(images_per_topic), num_columns, figsize=(17, 3*len(images_per_topic)))
            fig.suptitle(self.bag_folder)
            for i, (_, images) in enumerate(images_per_topic.items()):
                image_idxs = np.round(np.linspace(0,len(images)-1,5)).astype(int)
                for j in range(len(image_idxs)):
                    image_rgb = cv2.cvtColor(images[image_idxs[j]], cv2.COLOR_BGR2RGB)
                    ax[i,j].imshow(image_rgb)
                    ax[i,j].set_title(f"Topic: {'camera_' + str(i).zfill(len(str(len(images_per_topic))))}, Image {j+1}/{len(image_idxs)}")
            plt.show()
