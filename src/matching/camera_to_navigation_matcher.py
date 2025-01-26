#!/usr/bin/env python
"""
This script aligns the camera and navigation timestamps to
augment the camera metadata with the matched navigation data. 
It also plots certain camera trajectories and their image
boundaries projected onto the seabed for debugging purposes.
"""

# Import external libraries 
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
from pyproj import Transformer
import pandas as pd 
import numpy as np
import json 
import os

# Import internal scripts
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

class CameraToNavigationMatcher:
    """This class extracts sonar images from several XTF files and stores metadata information of every image."""

    def __init__(self):
        """"Constructor for the CameraToNavigationMatcher class. 
        It processes every XTF file, extracts the images and stores the metadata." """

        print("\nMATCHING NAVIGATION DATA TO CAMERA IMAGES...")

        # Initialise transformer from E/N to Lat/Lon
        self.transformer = Transformer.from_crs("EPSG:" + str(config.EPSG_CODE), "EPSG:4326")

        # Get folders of processed navigation and camera data 
        camera_processing_folder = os.path.join(config.PROCESS_FOLDER_PATH, "camera")
        self.navigation_folder = os.path.join(config.PROCESS_FOLDER_PATH, "navigation")

        # Iterate over files inside camera folder
        camera_metadata_file_paths = [] 
        for (root, _, file_names) in os.walk(camera_processing_folder, topdown=True):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                if file_name == "camera.csv":
                    camera_metadata_file_paths.append(file_path)

        # Check if metadata files are present on disk
        if len(camera_metadata_file_paths) == 0:
            print(f"No metadata present inside {camera_processing_folder}!")
            return

        # Iterate over files inside camera folder 
        for i, camera_metadata_file_path in enumerate(camera_metadata_file_paths):
            bag_folder, camera_metadata_file_name = camera_metadata_file_path.split("/")[-2:] 

            if config.WAIT_FOR_USER_INPUT:
                input("\nPress Enter to continue...")

            print(f"\nProcessing {i+1}/{len(camera_metadata_file_paths)}:  {camera_metadata_file_path}...")
            
            # Check if the current ROS Bag topic metadata was already added to the final metadata file 
            camera_with_nav_metadata_file_path = os.path.join(camera_processing_folder, bag_folder, "camera_with_nav.csv")
            if os.path.exists(camera_with_nav_metadata_file_path):
                print(f"Metadata file {camera_with_nav_metadata_file_path} already present on disk!")
                continue

            # Get camera info json file path
            topic_info_file_path = os.path.join(camera_processing_folder, bag_folder, "info.json")
            with open(topic_info_file_path) as json_file:
                topic_info = json.load(json_file)

            if len(topic_info) == 0:
                print(f"No topic information inside {topic_info_file_path}. Skipping...")
                continue 

            # Process camera csv file
            camera_with_nav_metadata_df = self.process_camera_csv(camera_metadata_file_path, topic_info)
            print("CSV processed!")

            # Save dataframe as csv 
            camera_with_nav_metadata_df.to_csv(camera_with_nav_metadata_file_path, index=False)
            print("Data saved to disk!")


    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PREPROCESSING FUNCTIONS                                                                       #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 


    def process_camera_csv(self, camera_metadata_file_path: str, 
                                 topic_info: Dict[str, Union[str, int, List[float], List[List[float]]]]) -> Optional[pd.DataFrame]:
        """Finds corresponding navigation data and merges it inside the camera metadata 

        Args:
            camera_metadata_file_path (str): Camera metadata file path 
            topic_info (Dict[str, str | int | List[float] | List[List[float]]]): Calibration data per topic inside ROS bag 

        Returns:
            pd.DataFrame: camera metadata with added navigation data 
        """

        # Match camera and navigation csv files 
        navigation_file_path = self.find_matching_navigation_file(camera_metadata_file_path)
        if navigation_file_path is None:
            return None 
        
        print("Navigation and camera timestamps matched!")
        
        # Load csv files as dataframes
        navigation_df = pd.read_csv(navigation_file_path)
        camera_df = pd.read_csv(camera_metadata_file_path)
        
        # Match timestamps
        navigation_timestamps = np.asarray(navigation_df["Timestamp_Seconds"])
        camera_timestamps = np.asarray(camera_df["Timestamp_Seconds"])        
        matching_navigation_idxs = self.match_timestamps(camera_timestamps, navigation_timestamps) 

        # Extract corresponding navigation data 
        navigation_data = navigation_df.drop(["Timestamp_Seconds"], axis=1).iloc[matching_navigation_idxs,:].reset_index(drop=True) 
       
        # Recalculate data for camera 
        updated_navigation_data = []
        valid_idxs = []
        for i in range(navigation_data.shape[0]):
            
            try:
                camera_index = camera_df["Camera_Index"].iloc[i]
                topic_data = topic_info["camera_" + str(camera_index)]
                row_data = navigation_data.iloc[i,:].to_numpy()

            except KeyError:
                continue 

            # Calculate SO(3) (NED coordinates) of camera wrt world
            T_base_to_camera = np.array(topic_data["extrinsics"])
            R_world_to_cam = common_utils.construct_R(*row_data[5:8]) @ T_base_to_camera[:3,:3]

            # Calculate SE(3) (ENU coordinates) of camera wrt world
            T_world_to_base = common_utils.construct_T(*row_data[2:4],row_data[-1],*row_data[5:8])
            R_enu_to_ned = np.array([[0,1,0],[1,0,0],[0,0,-1]])
            T_world_to_base[:3,:3] = R_enu_to_ned @ T_world_to_base[:3,:3] 
            T_world_to_cam = T_world_to_base @ T_base_to_camera

            # Calculate new data
            east, north, depth = T_world_to_cam[:3,3]
            roll, pitch, yaw = common_utils.rotation_matrix_to_euler_angles(R_world_to_cam)
            lat, lon = self.transformer.transform(east, north)
            altitude = row_data[-2] - (T_world_to_cam[2,3] - T_world_to_base[2,3])

            # Filter by altitude 
            if altitude <= config.MAX_IMAGE_HEIGHT:
                valid_idxs.append(i)

                # Calculate projections of image corners on sea-bed 
                image_corners_pixel =  [[0,0,1], [topic_data["width"],0,1], [0,topic_data["height"],1], [topic_data["width"],topic_data["height"],1]]
                image_corners_world = np.apply_along_axis(camera_utils.pixel_to_seabed, 1, image_corners_pixel, altitude, np.array(topic_data["intrinsics"]), T_world_to_cam)

                # Construct new row
                updated_row = [lat, lon, east, north, config.EPSG_CODE, roll, pitch, yaw, altitude, depth, *image_corners_world[:,:2].flatten().tolist()]
                updated_navigation_data.append(updated_row)
        
        # Combine with the camera data 
        additional_columns = ["Top_Left_East","Top_Left_North","Top_Right_East","Top_Right_North",
                              "Bottom_Left_East","Bottom_Left_North","Bottom_Right_East","Bottom_Right_North"]
        combined_df = pd.concat([camera_df.iloc[valid_idxs].reset_index(drop=True), pd.DataFrame(updated_navigation_data, columns=list(navigation_data.columns)+additional_columns)], axis=1)

        if config.DEBUG_MODE:
            self.debug_plots(camera_metadata_file_path.split("/")[-2], navigation_df, combined_df)

        print("Navigation and camera data merged!")       
        return combined_df


    def find_matching_navigation_file(self, camera_csv_file_path: str) -> Optional[str]:
        """Finds the navigation file path linked to the same ROS bag of the camera metadata file 

        Args:
            camera_csv_file_path (str): Camera metadat filepath 

        Returns:
            None | str: Navigation file path or None 
        """

        camera_csv_root_folder = camera_csv_file_path.split("/")[-2]
        for navigation_csv_file_name in os.listdir(self.navigation_folder):
            if navigation_csv_file_name.split(".csv")[0] == camera_csv_root_folder:
                return os.path.join(self.navigation_folder, navigation_csv_file_name)
        return None 
            

    def match_timestamps(self, timestamps1: np.ndarray, 
                               timestamps2: np.ndarray) -> np.ndarray:
        """Match two set of timestamps

        Args:
            timestamps1 (np.ndarray): First set of timestamps
            timestamps2 (np.ndarray): Second set of timestamps

        Returns:
            np.ndarray: Indices of second set of timestamps that match with the first set 
        """

        matching_timestamp2_idxs = []
        for timestamp in timestamps1:
            idx = np.argmin(np.abs(timestamps2 - timestamp))
            matching_timestamp2_idxs.append(idx)
        return np.asarray(matching_timestamp2_idxs)
    

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PLOTTING FUNCTIONS                                                                            #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 

    
    def debug_plots(self, title: str, 
                          navigation_df: pd.DataFrame, 
                          camera_df: pd.DataFrame) -> None:
        """Plots on matplotlib the camera trajectories, the angles over time, the altitude and depth, and the projection of the image boundaries on the sea-bed.

        Args:
            title (str): figure title
            navigation_df (pd.DataFrame): navigation data 
            camera_df (pd.DataFrame): camera images metadata 
        """

        fig, ax = plt.subplots(2,3,figsize=(15,8))
        fig.suptitle(title)

        initial_timestamp = navigation_df["Timestamp_Seconds"].iloc[0]

        # Plot the original navigation trajectory  camera trajectory and the 
        ax[0,0].plot(navigation_df["East"].to_numpy(), navigation_df["North"].to_numpy(), label="auv")
        ax[0,0].axis("equal")
        ax[0,0].set_xlabel("Easting [metres]")
        ax[0,0].set_ylabel("Northing [metres]")
        ax[0,0].scatter([navigation_df["East"].iloc[0]], [navigation_df["North"].iloc[0]], label="Start", c="orange")
        ax[0,0].scatter([navigation_df["East"].iloc[-1]], [navigation_df["North"].iloc[-1]], label="End", c="violet")

        # Plot the left camera trajectory 
        left_idxs = np.flatnonzero(np.core.defchararray.find(camera_df["Topic_Name"].to_numpy().astype(str),"left")!=-1)
        if left_idxs.size != 0:
            ax[0,0].scatter(camera_df["East"][left_idxs], camera_df["North"][left_idxs], c="red", label="left cam")
        
        # Plot the right camera trajectory 
        right_idxs = np.flatnonzero(np.core.defchararray.find(camera_df["Topic_Name"].to_numpy().astype(str),"right")!=-1)
        if right_idxs.size != 0:
            ax[0,0].scatter(camera_df["East"][right_idxs], camera_df["North"][right_idxs], c="green", label="right cam")

        # Plot left and right image boundaries 
        for side, idxs, colour in zip(["left", "right"], [left_idxs, right_idxs], ["red", "green"]):

            if idxs.size == 0:
                continue 

            for i, idx in enumerate(idxs):

                topleft = camera_df[["Top_Left_East","Top_Left_North"]].iloc[idx,:].tolist()
                topright = camera_df[["Top_Right_East","Top_Right_North"]].iloc[idx,:].tolist()
                bottomleft = camera_df[["Bottom_Left_East","Bottom_Left_North"]].iloc[idx,:].tolist()
                bottomright =camera_df[["Bottom_Right_East","Bottom_Right_North"]].iloc[idx,:].tolist()

                if i == 0:
                    ax[0,0].plot([topleft[0],topright[0]],[topleft[1],topright[1]], color=colour, label=side)
                else:
                    ax[0,0].plot([topleft[0],topright[0]],[topleft[1],topright[1]], color=colour)
                ax[0,0].plot([topright[0],bottomright[0]],[topright[1],bottomright[1]], color=colour)
                ax[0,0].plot([bottomleft[0],bottomright[0]],[bottomleft[1],bottomright[1]], color=colour)
                ax[0,0].plot([bottomleft[0],topleft[0]],[bottomleft[1],topleft[1]], color=colour)

        ax[0,0].axis('equal')
        ax[0,0].legend()

        # Plot the lat, lon alongside the original navigation trajectory 
        ax[1,0].plot(navigation_df["Longitude"].to_numpy(), navigation_df["Latitude"].to_numpy(), label="auv")
        ax[1,0].scatter(camera_df["Longitude"].to_numpy(), camera_df["Latitude"].to_numpy(), c="orange", label="camera")
        ax[1,0].axis("equal")
        ax[1,0].set_xlabel("Longitude [deg]")
        ax[1,0].set_ylabel("Latitude [deg]")
        ax[1,0].scatter([navigation_df["Longitude"].iloc[0]], [navigation_df["Latitude"].iloc[0]], label="Start", c="red")
        ax[1,0].scatter([navigation_df["Longitude"].iloc[-1]], [navigation_df["Latitude"].iloc[-1]], label="End", c="green")
        ax[1,0].legend()

        # Plot the roll, pitch and yaw of the original and camera trajectories 
        ax[0,1].plot(navigation_df["Timestamp_Seconds"].to_numpy() - initial_timestamp, navigation_df["Roll_Rad"].to_numpy() * 180.0 / np.pi, label="auv roll")
        ax[0,1].plot(navigation_df["Timestamp_Seconds"].to_numpy() - initial_timestamp, navigation_df["Pitch_Rad"].to_numpy() * 180.0 / np.pi, label="auv pitch")
        ax[0,1].plot(navigation_df["Timestamp_Seconds"].to_numpy() - initial_timestamp, navigation_df["Yaw_Rad"].to_numpy() * 180.0 / np.pi, label="auv yaw")
        ax[0,1].set_xlabel("Time [seconds]")
        ax[0,1].set_ylabel("Angle [degrees]")
        ax[0,1].legend()

        ax[1,1].scatter(camera_df["Timestamp_Seconds"][left_idxs] - initial_timestamp, camera_df["Roll_Rad"][left_idxs] * 180.0 / np.pi, s=10, label="left cam roll")
        ax[1,1].scatter(camera_df["Timestamp_Seconds"][right_idxs] - initial_timestamp, camera_df["Roll_Rad"][right_idxs] * 180.0 / np.pi, s=10, label="right cam roll")
        ax[1,1].scatter(camera_df["Timestamp_Seconds"][left_idxs] - initial_timestamp, camera_df["Pitch_Rad"][left_idxs] * 180.0 / np.pi, s=10, label="left cam pitch")
        ax[1,1].scatter(camera_df["Timestamp_Seconds"][right_idxs] - initial_timestamp, camera_df["Pitch_Rad"][right_idxs] * 180.0 / np.pi, s=10, label="right cam pitch")
        ax[1,1].scatter(camera_df["Timestamp_Seconds"][left_idxs] - initial_timestamp, camera_df["Yaw_Rad"][left_idxs] * 180.0 / np.pi, s=10, label="left cam yaw")
        ax[1,1].scatter(camera_df["Timestamp_Seconds"][right_idxs] - initial_timestamp, camera_df["Yaw_Rad"][right_idxs] * 180.0 / np.pi, s=10, label="right cam yaw")
        ax[1,1].set_xlabel("Time [seconds]")
        ax[1,1].set_ylabel("Angle [degrees]")
        ax[1,1].legend() 

        # Plot altitude and depth alongside original values
        ax[0,2].plot(navigation_df["Timestamp_Seconds"].to_numpy() - initial_timestamp, navigation_df["Altitude"].to_numpy(), label="auv")
        ax[0,2].scatter(camera_df["Timestamp_Seconds"][left_idxs] - initial_timestamp, camera_df["Altitude"][left_idxs], c="red", label="left cam")
        ax[0,2].scatter(camera_df["Timestamp_Seconds"][right_idxs] - initial_timestamp, camera_df["Altitude"][right_idxs], c="orange", label="right cam")
        ax[0,2].set_xlabel("Time [seconds]")
        ax[0,2].set_ylabel("Altitude [metres]")
        ax[0,2].legend()

        ax[1,2].plot(navigation_df["Timestamp_Seconds"].to_numpy() - initial_timestamp, navigation_df["Depth"].to_numpy(), label="auv")
        ax[1,2].scatter(camera_df["Timestamp_Seconds"][left_idxs] - initial_timestamp, camera_df["Depth"][left_idxs], c="red", label="left cam")
        ax[1,2].scatter(camera_df["Timestamp_Seconds"][right_idxs] - initial_timestamp, camera_df["Depth"][right_idxs], c="orange", label="right cam")
        ax[1,2].set_xlabel("Time [seconds]")
        ax[1,2].set_ylabel("Depth [metres]")
        ax[1,2].legend()
        
        plt.show()