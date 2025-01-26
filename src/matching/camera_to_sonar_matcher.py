#!/usr/bin/env python
"""
This script matches the camera and sonar images using the 
geographical coordinates of the projections of the image boundaries
on the sea-bed. It stores the matched images and their metadata.  
It also plots the matches between camera and sonar images for debugging purposes.
"""

# Import external libraries 
from pandas.errors import SettingWithCopyWarning
from typing import List, Tuple, Optional, Union
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt 
from shapely import intersects 
from shapely import make_valid
from shapely import area  
import geopandas as gpd 
import pandas as pd
import numpy as np 
import warnings
import json 
import os 

# Import internal scripts
from src.builder import XTFPatchBuilder, CameraImageBuilder, BuilderException
from src.utils import sonar_utils
from src import config

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

class CameraToSonarMatcher:
    """This class matches the sonar and camera images, then stores the images and the metadata of the matched ones."""

    def __init__(self):
        """Constructor for the CameraToSonarMatcher class. 
        It matches the sonar and camera images, saves the images and stores the metadata.
        """

        self.sonar_header =  ["XTF_File_Name","Patch_File_Path","Port_Starboard","Start_Ping_Index","End_Ping_Index",
                              "Start_Bin_Index","End_Bin_Index","EPSG_Code","Top_Left_East","Top_Left_North",
                              "Top_Right_East","Top_Right_North","Bottom_Left_East","Bottom_Left_North",
                              "Bottom_Right_East","Bottom_Right_North"]
        self.camera_header = ["ROS_Bag_File_Name","Sequence_Index","Camera_Index","Topic_Name","Timestamp_Seconds",
                              "Image_File_Path","Latitude","Longitude","East","North","EPSG_Code","Roll_Rad","Pitch_Rad",
                              "Yaw_Rad","Altitude","Depth","Top_Left_East","Top_Left_North","Top_Right_East","Top_Right_North",
                              "Bottom_Left_East","Bottom_Left_North","Bottom_Right_East","Bottom_Right_North"]

        pd.set_option('display.max_columns', None)
        warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
       
        # Get folders of processed/dataset sonar and camera data 
        self.camera_processing_folder = os.path.join(config.PROCESS_FOLDER_PATH, "camera")
        self.camera_dataset_folder = os.path.join(config.DATASET_FOLDER_PATH, "camera")
        self.sonar_processing_folder = os.path.join(config.PROCESS_FOLDER_PATH, "sonar")
        self.sonar_dataset_folder = os.path.join(config.DATASET_FOLDER_PATH, "sonar")

        if not os.path.exists(self.camera_dataset_folder):
            os.makedirs(self.camera_dataset_folder)
        if not os.path.exists(self.sonar_dataset_folder):
            os.makedirs(self.sonar_dataset_folder) 

        # Check if dataset metadata is already on disk 
        camera_dataset_metadata_file_path = os.path.join(self.camera_dataset_folder, "camera.csv")
        sonar_dataset_metadata_file_path = os.path.join(self.sonar_dataset_folder, "sonar.csv")
        matches_metadata_file_path = os.path.join(config.DATASET_FOLDER_PATH, "correspondences.json")

        # Reset counters, remove patches/images not present in metadata and retrieve already processed xtfs 
        self.correct_inconsistencies(camera_dataset_metadata_file_path, sonar_dataset_metadata_file_path)
        self.reset_counters(camera_dataset_metadata_file_path, sonar_dataset_metadata_file_path)
        processed_xtfs = self.retrieve_processed_xtfs(sonar_dataset_metadata_file_path)

        # Get metadata files 
        metadata_file_paths = self.get_metadata()
        if metadata_file_paths is None:
            return 
        else:
            camera_metadata_file_paths, sonar_metadata_file_paths = metadata_file_paths 
        
        self.sonar_metadata_df = None if not os.path.exists(sonar_dataset_metadata_file_path) else pd.read_csv(sonar_dataset_metadata_file_path)
        self.camera_metadata_df = None if not os.path.exists(camera_dataset_metadata_file_path) else pd.read_csv(camera_dataset_metadata_file_path)
        
        for i, sonar_metadata_file_path in enumerate(sonar_metadata_file_paths): 

            if config.WAIT_FOR_USER_INPUT:
                input("\nPress Enter to continue...")
            
            print(f"\nProcessing {i+1}/{len(sonar_metadata_file_paths)}:  {sonar_metadata_file_path}...")

            self.sonar_metadata_rows = pd.DataFrame(columns=self.sonar_header)
            self.camera_metadata_rows = pd.DataFrame(columns=self.camera_header)
            self.patches_on_disk = []
            self.images_on_disk = []
            self.correspondences = {}
  
            self.match_images_to_patches(sonar_metadata_file_path, camera_metadata_file_paths, processed_xtfs) 

            # Visualisation 
            if config.DEBUG_MODE:
                print("Debug: Not implemented yet!")  

            # Build dataset 
            if self.sonar_metadata_rows.empty or self.camera_metadata_rows.empty:
                print("No matched data. Skipping...")
                continue 

            # Save patches and images
            if config.BUILD_DATASET:
                try:
                    builder = XTFPatchBuilder(self.sonar_metadata_rows, image_locations_on_disk=self.patches_on_disk)
                except BuilderException as e:
                    print(f"{e} Skipping...")
            
                try:
                    builder = CameraImageBuilder(self.camera_metadata_rows, image_locations_on_disk=self.images_on_disk)
                except BuilderException as e:
                    print(f"{e} Skipping...")

            # Create dataframe and save to disk as .metadata
            if not os.path.exists(sonar_dataset_metadata_file_path):
                self.sonar_metadata_rows.to_csv(sonar_dataset_metadata_file_path, index=False)
            else:
                self.sonar_metadata_rows.to_csv(sonar_dataset_metadata_file_path, mode='a', index=False, header=False)
                
            if not os.path.exists(camera_dataset_metadata_file_path):
                self.camera_metadata_rows.to_csv(camera_dataset_metadata_file_path, index=False)
            else:
                self.camera_metadata_rows.to_csv(camera_dataset_metadata_file_path, mode='a', index=False, header=False)                

            if not os.path.exists(matches_metadata_file_path):
                # Create json 
                with open(matches_metadata_file_path, 'w') as f:
                    json.dump(self.correspondences, f, indent=4) 
            else:
                # Append to json
                with open(matches_metadata_file_path , "r") as f:
                    data = json.loads(f.read())
                data.update(self.correspondences)
                with open(matches_metadata_file_path , "w") as f:
                    json.dump(data, f, indent=4)

            print("Metadata saved to disk!")

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PREPROCESSING FUNCTIONS                                                                       #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 


    def generate_metadata(self, sonar_metadata_df: pd.DataFrame, camera_metadata_df: pd.DataFrame, matching_idxs: np.ndarray) -> None:
        """Create metadata information for the matching sonar and camera images to be stored in the dataset 

        Args:
            sonar_metadata_df (pd.DataFrame): Sonar metadata (from preprocessed data)
            camera_metadata_df (pd.DataFrame): Camera metadata (from preprocessed data)
            matching_idxs (np.ndarray): Matching indices betweent the sonar and camera images
        """

        for i, patch_idx in enumerate(np.unique(matching_idxs[:,0])):
            image_idxs = matching_idxs[matching_idxs[:,0] == patch_idx,1]

            # Check if patch is already on disk
            patch_file_path = sonar_metadata_df["Patch_File_Path"].iloc[patch_idx]
            self.patches_on_disk.append(patch_file_path) if os.path.exists(patch_file_path) else self.patches_on_disk.append(np.nan)

            # Patch metadata 
            patch_metadata_row = sonar_metadata_df.iloc[patch_idx,:]
            patch_file_path = os.path.join(self.sonar_dataset_folder, str(self.patch_counter).zfill(6) + ".png")
            patch_metadata_row["Patch_File_Path"] = patch_file_path
            self.sonar_metadata_rows.loc[len(self.sonar_metadata_rows)] = patch_metadata_row
            self.patch_counter += 1

            # Save camera images   
            image_file_paths = []
            for image_idx in image_idxs:
                
                # Check if image is already on disk
                image_file_path = camera_metadata_df["Image_File_Path"].iloc[image_idx]
                self.images_on_disk.append(image_file_path) if os.path.exists(image_file_path) else self.images_on_disk.append(np.nan)

                # Image metadata                
                image_metadata_row = camera_metadata_df.iloc[image_idx,:]
                image_file_path =  os.path.join(self.camera_dataset_folder, str(self.image_counter).zfill(6) + ".png")
                image_metadata_row["Image_File_Path"] = image_file_path

                # Check if image is already inside dataset 
                existing_image_file_path = self.check_if_image_exists(image_metadata_row)
                if existing_image_file_path is None:
                    self.camera_metadata_rows.loc[len(self.camera_metadata_rows)] = image_metadata_row
                    self.image_counter += 1
                else:
                    image_file_path = existing_image_file_path
                
                image_file_paths.append(image_file_path)
            
            self.correspondences[patch_file_path] = image_file_paths


    def intersection_by_area(self, patch_polygons: List[Polygon], 
                                   image_polygons: List[Polygon]) -> np.ndarray:
        """Intersect the two sets of polygons using as threshold the percent overlap 

        Args:
            patch_polygons (List[Polygon]): List of polygons made from sonar image corners
            image_polygons (List[Polygon]): List of polygons made from camera image corners

        Returns:
            np.ndarray: Indices of matching polygons between the two sets 
        """

        # Convert list of geometries to GeoSeries
        patch_gdf = gpd.GeoDataFrame({"geometry": patch_polygons, "patch_idxs": [i for i in range(len(patch_polygons))]})
        image_gdf = gpd.GeoDataFrame({"geometry": image_polygons, "image_idxs": [i for i in range(len(image_polygons))]})

        # Intersect GeoSeries 
        intersection_gdf = gpd.overlay(patch_gdf, image_gdf, how='intersection')

        # Filter intersections by area
        intersection_gdf["intersection_area"] = intersection_gdf["geometry"].apply(lambda x: area(x))
        intersection_gdf["image_area"] = intersection_gdf["image_idxs"].apply(lambda x: area(image_polygons[x]))
        matches = (intersection_gdf["intersection_area"] >= config.PERCENT_IMAGE_OVERLAP * intersection_gdf["image_area"]).to_numpy()

        # Return matching indices 
        matching_idxs = intersection_gdf[["patch_idxs","image_idxs"]].iloc[matches,:].to_numpy() 
        return matching_idxs
        

    def create_polygons(self, metadata_df: pd.DataFrame) -> Tuple[List[Polygon], Polygon]:
        """Create list of polygons from image corners (rectangle/square) and a union of all of these polygons 

        Args:
            metadata_df (pd.DataFrame): Metadata 

        Returns:
            Tuple[List[Polygon], Polygon]: List of polygons made from image perimeters and union of all of these polygons
        """

        # Create Polygon's from each image/patch
        polygons = metadata_df.apply(lambda row: Polygon([[row["Top_Left_East"],row["Top_Left_North"]],
                                                          [row["Top_Right_East"],row["Top_Right_North"]],
                                                          [row["Bottom_Right_East"],row["Bottom_Right_North"]],
                                                          [row["Bottom_Left_East"],row["Bottom_Left_North"]]]), axis=1)

        # Transform non valid polygons to valid ones 
        for k in range(len(polygons)):
            if not polygons[k].is_valid:
                polygons[k] = make_valid(polygons[k])

        polygons_union = unary_union(polygons) 

        return polygons.tolist(), polygons_union
    

    def reset_counters(self, camera_dataset_metadata_file_path: str, 
                             sonar_dataset_metadata_file_path: str) -> None:
        """Use the metadata information to set the correct counters for the new set of camera and sonar images 

        Args:
            camera_dataset_metadata_file_path (str): Camera metadata (of dataset) file path 
            sonar_dataset_metadata_file_path (str): Sonar metadata (of dataset) file path
        """

        # Set the image counter 
        self.image_counter = 0 
        if os.path.exists(camera_dataset_metadata_file_path):
            camera_metadata_df = pd.read_csv(camera_dataset_metadata_file_path)
            self.image_counter = int(camera_metadata_df["Image_File_Path"].iloc[-1].split("/")[-1].split(".png")[0]) + 1

        #  Set the patch counter 
        self.patch_counter = 0
        if os.path.exists(sonar_dataset_metadata_file_path):
            sonar_metadata_df = pd.read_csv(sonar_dataset_metadata_file_path)
            self.patch_counter = int(sonar_metadata_df["Patch_File_Path"].iloc[-1].split("/")[-1].split(".png")[0]) + 1

    def retrieve_processed_xtfs(self, sonar_dataset_metadata_file_path: str) -> List[str]:
        """Find the XTF files that were already processed 

        Args:
            sonar_dataset_metadata_file_path (str): Sonar metadata (of dataset) file path

        Returns:
            List[str]: list of already processed XTF files
        """

        processed_xtfs = []
        if os.path.exists(sonar_dataset_metadata_file_path):
            sonar_metadata_df = pd.read_csv(sonar_dataset_metadata_file_path)
            processed_xtfs = np.unique(sonar_metadata_df["XTF_File_Name"])
        return processed_xtfs

    def correct_inconsistencies(self, camera_dataset_metadata_file_path: str, 
                                      sonar_dataset_metadata_file_path: str) -> None:
        """Remove camera and sonar images that are present in the directory but not in the metadata 

        Args:
            camera_dataset_metadata_file_path (str): Camera metadata (of dataset) file path 
            sonar_dataset_metadata_file_path (str): Sonar metadata (of dataset) file path
        """

        # Remove patches that do not appear in metadata file
        if os.path.exists(sonar_dataset_metadata_file_path):
            sonar_metadata_df = pd.read_csv(sonar_dataset_metadata_file_path)

        for patch_file_name in os.listdir(self.sonar_dataset_folder):
            if patch_file_name.endswith(".png"):
                patch_file_path = os.path.join(self.sonar_dataset_folder, patch_file_name)
                if not os.path.exists(sonar_dataset_metadata_file_path) or patch_file_path not in sonar_metadata_df["Patch_File_Path"].values:
                    os.remove(patch_file_path)
        
        # Remove images that do not appear in metadata file
        if os.path.exists(camera_dataset_metadata_file_path):
            camera_metadata_df = pd.read_csv(camera_dataset_metadata_file_path)

        for image_file_name in os.listdir(self.camera_dataset_folder):
            if image_file_name.endswith(".png"):
                image_file_path = os.path.join(self.camera_dataset_folder, image_file_name)
                if not os.path.exists(camera_dataset_metadata_file_path) or image_file_path not in camera_metadata_df["Image_File_Path"].values:
                    os.remove(image_file_path)
        
    
    def get_metadata(self) -> Optional[Tuple[List[str], List[str]]]:
        """Find sonar and camera metadata file paths that are already present inside the directory  

        Returns:
            Tuple[List[str], List[str]] | None: List of metadata file paths found in the directory 
        """

        # Iterate over files inside processing folder 
        camera_metadata_file_paths = [] 
        sonar_metadata_file_paths = [] 
        for (root, _, file_names) in os.walk(config.PROCESS_FOLDER_PATH, topdown=True):
            for file_name in file_names:
                if file_name == "camera_with_nav.csv":
                    camera_metadata_file_paths.append(os.path.join(root, file_name))
                if file_name == "sonar.csv":
                    sonar_metadata_file_paths.append(os.path.join(root, file_name))
        
        if len(camera_metadata_file_paths) == 0 or len(sonar_metadata_file_paths) == 0:
            print(f"No metadata present inside {self.camera_processing_folder} and/or {self.sonar_processing_folder}!")
            return None 
        else:
            return camera_metadata_file_paths, sonar_metadata_file_paths
        
    
    def match_images_to_patches(self, sonar_metadata_file_path: str, 
                                      camera_metadata_file_paths: List[str], 
                                      processed_xtfs: List[str]) -> None:
        """Match camera images from all ROS bags to sonar images from a single XTF file."

        Args:
            sonar_metadata_file_path (str): Sonar metadata (of XTF file) file path 
            camera_metadata_file_paths (List[str]): Camera metadata (of ROS bags) file paths  
            processed_xtfs (List[str]): List of already processed XTF files 
        """

        sonar_metadata_df = pd.read_csv(sonar_metadata_file_path)
            
        # Skip if XTF was already processed
        if sonar_metadata_df["XTF_File_Name"].iloc[0] in processed_xtfs:
            print(f"Metadata file {sonar_metadata_file_path} was already processed! Skipping...")
            return  

        # Create Polygon's from sonar metadata
        patch_polygons, patch_polygons_union = self.create_polygons(sonar_metadata_df)

        for camera_metadata_file_path in camera_metadata_file_paths:
                
            # Load metadata from disk into DataFrame
            camera_metadata_df = pd.read_csv(camera_metadata_file_path) 
            if camera_metadata_df.empty:
                continue

            # Create Polygon's from camera metadata 
            image_polygons, image_polygons_union = self.create_polygons(camera_metadata_df)

            # Process sonar and camera metadata 
            if not intersects(image_polygons_union, patch_polygons_union):
                continue 
            else:
                # Find the corresponding matches 
                matching_idxs = self.intersection_by_area(patch_polygons, image_polygons)
                self.generate_metadata(sonar_metadata_df, camera_metadata_df, matching_idxs)

                # Plot both geometries 
                if config.DEBUG_MODE:
                    xtf_file_path = os.path.join(config.XTF_FOLDER_PATH, sonar_metadata_df["XTF_File_Name"].iloc[0])
                    patch_polygons_matches = [patch_polygons[idx] for idx in np.unique(matching_idxs[:,0])]
                    image_polygons_matches = [image_polygons[idx] for idx in np.unique(matching_idxs[:,1])]
                    self.plot_matches(camera_metadata_df, xtf_file_path, image_polygons_matches, patch_polygons_matches)

    
    def check_if_image_exists(self, row: List[Union[str, int]]) -> Optional[str]:
        """Finds the camera image file path from the metadata information 

        Args:
            row (List[str  |  int]): camera image metadata 

        Returns:
            str | None: image file if it already exists 
        """

        # Find the camera image file path from previously stored metadata
        if self.camera_metadata_df is not None:

            matches = (self.camera_metadata_df["ROS_Bag_File_Name"] == row["ROS_Bag_File_Name"]) & \
               (self.camera_metadata_df["Sequence_Index"] == row["Sequence_Index"]) & \
               (self.camera_metadata_df["Topic_Name"] == row["Topic_Name"])

            if np.any(matches):
                image_file_path = self.camera_metadata_df["Image_File_Path"][np.nonzero(matches)[0][0]]
                return image_file_path
            else:
                return None
        
        # Find the camera image file path from current metadata 
        if not self.camera_metadata_rows.empty:
            
            matches = (self.camera_metadata_rows["ROS_Bag_File_Name"] == row["ROS_Bag_File_Name"]) & \
               (self.camera_metadata_rows["Sequence_Index"] == row["Sequence_Index"]) & \
               (self.camera_metadata_rows["Topic_Name"] == row["Topic_Name"])

            if np.any(matches):
                image_file_path = self.camera_metadata_rows["Image_File_Path"][np.nonzero(matches)[0][0]]
                return image_file_path
            else:
                return None
            
        return None 


    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PLOTTING FUNCTIONS                                                                            #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 

    def plot_matches(self, camera_metadata_df: pd.DataFrame,
                           sonar_xtf_file_path: str,
                           camera_polygons: List[Polygon],
                           sonar_polygons: List[Polygon]) -> None:
        """Plots on matplotlib the camera and sonar trajectories and the geographical coordinates of the matched images (both sonar and camera). 

        Args:
            camera_metadata_df (pd.DataFrame): Camera metadata
            sonar_xtf_file_path (str): XTF file path
            camera_polygons (List[Polygon]): Polygons of camera image projections on sea-bed
            sonar_polygons (List[Polygon]): Polygons of sonar image projections on sea-bed
        """
        _, ax = plt.subplots()

        # Plot SSS trajectory
        pings = sonar_utils.load_xtf(sonar_xtf_file_path) 
        _, traj = sonar_utils.calculate_swath_positions(pings)
        ax.plot(traj[:,0], traj[:,1], "red", label="sonar trajectory")

        # Plot camera trajectory 
        rosbag_name =  camera_metadata_df["ROS_Bag_File_Name"].iloc[0] 
        ax.plot(camera_metadata_df["East"].tolist(), camera_metadata_df["North"].tolist(), "blue", label="camera trajectory")

        for i, camera_polygon in enumerate(camera_polygons):
            xs, ys = camera_polygon.exterior.xy
            if i == 0:
                ax.plot(xs, ys, color="blue", label="camera boundaries") 
            else:
                ax.plot(xs, ys, color="blue")  

        for i, sonar_polygon in enumerate(sonar_polygons):
            xs, ys = sonar_polygon.exterior.xy
            if i == 0:
                ax.plot(xs, ys, color="red", label="sonar boundaries") 
            else:
                ax.plot(xs, ys, linewidth=3, color="red")  

        # Axis scale and title
        ax.axis("equal")
        ax.set_title(f"Matches between XTF file: {sonar_xtf_file_path.split('/')[-1]}\nand ROS bag: {rosbag_name}")

        # Plot figure 
        plt.legend()
        plt.show()  
