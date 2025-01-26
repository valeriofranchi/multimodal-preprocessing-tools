#!/usr/bin/env python
"""
This script reads a ROS bag either from a ROS1 or ROS2 architecture. 
"""

# Import external libraries 
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.rosbag1 import Reader as ROS1Reader
from rosbags.rosbag2 import Reader as ROS2Reader
from typing import List, Optional
import os 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

class BagReaderException(Exception):
    """Custom exception for BagReader class 

    Args:
        Exception (_type_): inherits from base Exception class 
    """
    def __init__(self, message: str):
        """Constructor for the BagReaderException class. 
        It prints out an exception message 

        Args:
            message (str): Exception message
        """
        super().__init__(f"BagReaderException: {message}")

class BagReader:
    """This class extracts ROS messages from ROS1 or ROS2 bags."""

    def __init__(self, bag_file_path: str, 
                       using_ros2: bool):
        """Constructor for the BagReader class. 
        Checks whether the ROS Bag presents any issues and initialises a ROS1/ROS2 reader.

        Args:
            bag_file_path (str): ROS bag file path 
            using_ros2 (bool): ROS2(True) or ROS1(False) bag 

        Raises:
            FileNotFoundError: File not found on disk 
            TypeError: File is not a ROS2 bag file
            TypeError: File is not a ROS1 bag file
            BagReaderException: File is a ROS1 Bag and system is running ROS2
            BagReaderException: File is a ROS2 Bag and system is running ROS1
        """

        self.bag_file_name = bag_file_path.split("/")[-1]

        # Determine if the Bag is a ROS1 Bag or a ROS2 Bag 
        if os.path.isfile(bag_file_path):
            self.is_ros2_bag = False 
        elif os.path.isdir(bag_file_path):
            self.is_ros2_bag = True 

        # Check if file path is correct
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError("Invalid Path", bag_file_path)
        
        # Check if file path is a ROS Bag before initialising the reader object
        if self.is_ros2_bag:
            if not os.path.exists(os.path.join(bag_file_path, bag_file_path.split("/")[-1] + ".db3")) or \
               not os.path.exists(os.path.join(bag_file_path, "metadata.yaml")):
                raise TypeError("Invalid File", bag_file_path)
            self.reader = ROS2Reader(bag_file_path)
        else:
            if not bag_file_path.endswith('.bag'):
                raise TypeError("Invalid File", bag_file_path)
            self.reader = ROS1Reader(bag_file_path)

        # Check if the ROS version and the Bag type are compatible
        if using_ros2 and not self.is_ros2_bag:
            raise BagReaderException(f"File {bag_file_path} is a ROS1 Bag and system is running ROS2!")
        
        if not using_ros2 and self.is_ros2_bag:
            raise BagReaderException(f"File {bag_file_path} is a ROS2 Bag and system is running ROS1!")
        
    def load_bag(self):
        """Opens the ROS bag
        """

        # Open the ROS Bag 
        self.reader.open() 
        self.typestore = get_typestore(Stores.EMPTY) 

        # Register the ROS messages present inside the ROS Bag 
        bag_msg_types = {}  
        for connection in self.reader.connections:
            bag_msg_types.update(get_types_from_msg(connection.msgdef,connection.msgtype))
        self.typestore.register(bag_msg_types)

    def message_count(self, topic: str) -> int:
        """Counts the numbers of message from a specific topic inside the ROS bag 

        Args:
            topic (str): Name of topic 

        Raises:
            BagReaderException: Topic is not present inside the ROS bag 

        Returns:
            int: Number of messages 
        """

        for connection in self.reader.connections:
            if topic == connection.topic:
                return connection.msgcount
        raise BagReaderException(f"Topic {topic} not present inside the ROS Bag!")
    
    def check_if_topic_in_bag(self, topic: str) -> bool:
        """Checks if the topic is present inside the ROS bag 

        Args:
            topic (str): Name of topic 

        Returns:
            bool: Topic is inside the bag (True) or not (False)
        """

        for connection in self.reader.connections:
            if connection.topic == topic:
                return True
        return False
    
    def check_if_ros2_bag(self) -> bool:
        """Checks if it is a ROS1 or ROS2 bag 

        Returns:
            bool: ROS2 (True) or ROS1 (False) bag 
        """

        return self.is_ros2_bag
    
    def get_bag_file_name(self) -> str:
        """Returns ROS bag file name 

        Returns:
            str: ROS bag file name 
        """

        return self.bag_file_name

    def extract_messages(self, topic_names: Optional[List[str]] = None):
        """Extracts the messages from every topic or from specific topics 

        Args:
            topic_names (List[str] | None, optional): Names of topics. Defaults to None.

        Yields:
            _type_: Current topic name and ROS message 
        """

        # Filter topics 
        if topic_names is None:
            connections = self.reader.connections
        else:
            connections = [x for x in self.reader.connections if x.topic in topic_names]

        # Iterate over ROS Bag messages
        for connection, _, raw_data in self.reader.messages(connections=connections):
            
            # Process topics from parameter 
            if connection.topic in topic_names:
                # Deserialise ROS Bag messasge
                if self.is_ros2_bag:
                    msg = self.typestore.deserialize_cdr(raw_data, connection.msgtype)
                else:
                    msg = self.typestore.deserialize_ros1(raw_data, connection.msgtype)

                yield connection.topic, msg 

        # Close reader after processing all the messages 
        self.reader.close() 
