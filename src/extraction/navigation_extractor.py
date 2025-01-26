#!/usr/bin/env python
"""
This script performs navigation data extraction from ROS bags. 
It also plots the extracted data for debugging purposes.
"""

# Import external libraries 
from cola2_msgs.msg import NavSts
import matplotlib.pyplot as plt
from typing import List, Union
from pyproj import Proj, CRS
import pandas as pd 
import numpy as np
import os 

# Import internal scripts
from src.bag_reader import BagReader, BagReaderException
from src import config

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

class NavigationExtractor:
    """This class extracts navigation data from several ROS bags and stores it. """
    
    def __init__(self):
        """Constructor for the NavigationExtractor class. 
        It processes every ROS Bag, extracts the navigation data and stores it."""

        print("\nEXTRACTING NAVIGATION...")

        # Create subfolder
        self.nav_folder = os.path.join(config.PROCESS_FOLDER_PATH, "navigation")
        if not os.path.exists(self.nav_folder):
            os.makedirs(self.nav_folder)

        # Define metadata file header 
        header = ["Timestamp_Seconds","Latitude","Longitude",
                  "East","North","EPSG_Code","Roll_Rad","Pitch_Rad","Yaw_Rad",
                  "Altitude","Depth"]

        # Process ROS Bags  
        bag_file_names = os.listdir(config.ROSBAG_FOLDER_PATH)
        for i, bag_file_name in enumerate(bag_file_names):

            if config.WAIT_FOR_USER_INPUT:
                input("\nPress Enter to continue...")

            # Define metadata file path 
            bag_file_path = os.path.join(config.ROSBAG_FOLDER_PATH, bag_file_name)
            nav_csv_file_name = bag_file_name.split("/")[-1].split(".bag")[0] + ".csv"

            print(f"\nProcessing {i+1}/{len(bag_file_names)}:  {bag_file_path}...")

            # Check if navigation metadata is already present on the disk 
            nav_csv_file_path = os.path.join(self.nav_folder, nav_csv_file_name) 
            if os.path.exists(nav_csv_file_path):
                print(f"File {nav_csv_file_path} already present! Skipping...")
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

            # Check if navigation topic is in ROS Bag
            if not reader.check_if_topic_in_bag(config.NAVIGATION_TOPIC):
                print("Invalid navigation topic. Skipping bag...")
                continue      

            # Process bag 
            navigation_csv_rows = self.process_bag(reader, config.NAVIGATION_TOPIC)
            print("Bag processed!")

            if config.DEBUG_MODE:
               self.debug_plots(navigation_csv_rows)

            # Create dataframe and save to disk as .csv
            nav_df = pd.DataFrame(navigation_csv_rows, columns=header)
            nav_df.to_csv(nav_csv_file_path, index=False)

            if len(navigation_csv_rows) > 0:
                print("Metadata saved to disk!")
            else:
                print("No information!")

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PREPROCESSING FUNCTIONS                                                                       #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 


    def process_bag(self, reader: BagReader, 
                          navigation_topic: str) -> List[List[Union[int, float]]]:
        """Processes the rosbag, extracts the navigation data and stores it 

        Args:
            reader (BagReader): BagReader object
            navigation_topic (str): Navigation topic

        Returns:
            List[List[int | float]]: Navigation data per ROS message 
        """

        navigation_csv_rows = []

        # Iterate over messages 
        message_gen = reader.extract_messages([navigation_topic])
        while True:
            try:
                # Get message, extract navigation data and append it to list
                _, msg = next(message_gen)
                data = self.extract_navigation(msg)
                navigation_csv_rows.append(data)
            except StopIteration:
                break

        return navigation_csv_rows

    def extract_navigation(self, msg: NavSts) -> List[Union[int, float]]:
        """Extracts navigation data from NavSts ROS message 

        Args:
            msg (NavSts): ROS NavSts message 

        Returns:
            List[int | float]: Navigatin data for current ROS message
        """
        
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9  
        latitude = msg.global_position.latitude
        longitude = msg.global_position.longitude

        # Convert lat/lon to east/north
        lonlat2EN = Proj(CRS.from_epsg(config.EPSG_CODE), preserve_units=False)
        easting, northing = lonlat2EN(longitude, latitude)

        roll_radians = msg.orientation.roll
        pitch_radians = msg.orientation.pitch
        yaw_radians = msg.orientation.yaw

        altitude = msg.altitude
        depth = msg.position.depth

        return [timestamp,  latitude, longitude, easting, northing, config.EPSG_CODE, roll_radians, pitch_radians, yaw_radians, altitude, depth]


    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PLOTTING FUNCTIONS                                                                            #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 

    def debug_plots(self, navigation_data: List[List[Union[int, float]]]) -> None:
        """Plots the navigation data on a matplotlib figure. 

        Args:
            navigation_data (List[List[int  |  float]]): Navigation data per ROS message 
        """

        data = np.asarray(navigation_data).astype(np.float64)
        _, ax = plt.subplots(1,3,figsize=(12,5))
                
        # Plot the trajectory (east, north), the altitudes (metres) and and the RPY (degrees)
        ax[0].plot(data[:,3], data[:,4], label="trajectory")
        ax[0].axis("equal")
        ax[0].set_title("AUV Trajectory")
        ax[0].set_xlabel("Easting [metres]")
        ax[0].set_ylabel("Northing [metres]")
        ax[0].scatter([data[0,3]], [data[0,4]], label="Start", c="red")
        ax[0].scatter([data[-1,3]], [data[-1,4]], label="End", c="green")
        ax[0].legend()

        ax[1].plot(data[:,0]-data[0,0], data[:,-2])
        ax[1].set_title("Altitude over time")
        ax[1].set_xlabel("Time [seconds]")
        ax[1].set_ylabel("Altitude [metres]")

        ax[2].plot(data[:,0]-data[0,0], data[:,6] * 180.0 / np.pi, label="roll")
        ax[2].plot(data[:,0]-data[0,0], data[:,7] * 180.0 / np.pi, label="pitch")
        ax[2].plot(data[:,0]-data[0,0], data[:,8] * 180.0 / np.pi, label="yaw")
        ax[2].set_title("RPY over time")
        ax[2].set_xlabel("Time [seconds]")
        ax[2].set_ylabel("Angle [degrees]")
        ax[2].legend() 

        plt.show()