#!/usr/bin/env python
"""
This script extracts pings from the XTF files. 
It creates the sonar images based on configuration parameters, 
and it plots certain extracted images for debugging purposes.
"""

# Import external libraries 
from typing import List, Any
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
import os 

# Import internal scripts
from src.builder import XTFPatchBuilder, BuilderException
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

class XTFPatchExtractor:
    """This class extracts sonar images from several XTF files and stores metadata information of every image. """

    def __init__(self):
        """Constructor for the XTFPatchExtractor class. 
        It processes every XTF file, creates the sonar images and stores the metadata."""

        print("\nEXTRACTING PATCHES...")

        # Create subfolder
        self.sonar_folder = os.path.join(config.PROCESS_FOLDER_PATH, "sonar")
        if not os.path.exists(self.sonar_folder):
            os.makedirs(self.sonar_folder)
        
        # Define metadata header 
        self.header = ["XTF_File_Name","Patch_File_Path","Port_Starboard","Start_Ping_Index","End_Ping_Index",
                       "Start_Bin_Index","End_Bin_Index","EPSG_Code","Top_Left_East","Top_Left_North",
                       "Top_Right_East","Top_Right_North","Bottom_Left_East","Bottom_Left_North",
                       "Bottom_Right_East","Bottom_Right_North"]
            
        # Process XTFs 
        xtf_file_names = os.listdir(config.XTF_FOLDER_PATH)
        for i, xtf_file_name in enumerate(xtf_file_names):

            if config.WAIT_FOR_USER_INPUT:
                input("\nPress Enter to continue...")

            # Reset 
            self.patch_counter = 0
            self.sonar_data_rows = []

            # Load XTF pings
            xtf_folder = os.path.join(self.sonar_folder, xtf_file_name.split(".xtf")[0])
            xtf_file_path = os.path.join(config.XTF_FOLDER_PATH, xtf_file_name)
            self.patches_folder = os.path.join(xtf_folder, "patches")
            print(f"\nProcessing {i+1}/{len(xtf_file_names)}:  {xtf_file_path}...")

            # Check if sonar metadata already present on disk 
            sonar_metadata_file_path = os.path.join(xtf_folder, "sonar.csv")
            if os.path.exists(xtf_folder) and os.path.exists(sonar_metadata_file_path):
                print(f"Folder {xtf_folder} already present!")
                continue

            # Load XTF data
            xtf_pings = sonar_utils.load_xtf(xtf_file_path)

            if len(xtf_pings) < config.PATCH_HEIGHT:
                print(f"Too few pings ({len(xtf_pings)})! Skipping...")
                continue 
            
            swaths, trajectory = sonar_utils.calculate_swath_positions(xtf_pings)
            waterfall = sonar_utils.calculate_waterfall(xtf_pings)

            # Process xtf pings
            self.process_xtf(xtf_file_name, xtf_pings, swaths, waterfall)
            print("XTF processed!")                
        
            # Create xtf folder if it does not exist 
            if not os.path.exists(xtf_folder):
                os.makedirs(xtf_folder)

            # Save progress to disk 
            sonar_df = pd.DataFrame(self.sonar_data_rows, columns=self.header)
            if len(self.sonar_data_rows) > 0:

                if config.DEBUG_MODE:
                    self.plot_xtf_summary(trajectory)

                if config.SAVE_PATCHES:
                    try:
                        XTFPatchBuilder(sonar_df)
                    except BuilderException as e:
                        print(f"{e} Skipping...")
            else:
                print("No information!")
            
            # Create dataframe and save to disk as .metadata
            sonar_df.to_csv(sonar_metadata_file_path, index=False)

            if len(self.sonar_data_rows) > 0:
                print("Metadata saved to disk!")

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PREPROCESSING FUNCTIONS                                                                       #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 

    
    def process_xtf(self, xtf_file_name: str, 
                          xtf_pings: List[Any], 
                          swaths: np.ndarray, 
                          waterfall: np.ndarray) -> None:
        """Processes the XTF file, creates the sonar images and extracts metadata information.

        Args:
            xtf_file_name (str): XTF file path 
            xtf_pings (List[Any]): XTF ping data
            swaths (np.ndarray): UTM coordinates of each pixel inside the waterfall
            waterfall (np.ndarray): Waterfall image of the XTF file 
        """

        num_pings = len(xtf_pings)

        # Calculate start and end pings indices for batches
        step = config.PATCH_HEIGHT * 4
        self.waterfall_shape = waterfall.shape

        # Process data in batches
        for i in range(0, num_pings, step):
            idx = i // step
            if config.DEBUG_MODE:
                print(f"Processing ping {i} to ping {max(num_pings, i + step)} ({idx + 1}/{num_pings // step + 1})")
            ping_idx = slice(i, i + step)

            # Get indices of bins outside the blind zone
            port_idx, stbd_idx = sonar_utils.calculate_blind_zone_indices(xtf_pings[ping_idx])

            # Create patches for the port and starboard data 
            self.create_patches(swaths[ping_idx, port_idx], waterfall[ping_idx, port_idx], xtf_file_name, i, True)
            self.create_patches(swaths[ping_idx, stbd_idx], waterfall[ping_idx, stbd_idx], xtf_file_name, i, False)

            # Visualise the first waterfall (port and starboard)
            if config.DEBUG_MODE and i == 0:
                self.plot_waterfall(waterfall, ping_idx, port_idx, stbd_idx)

        if config.DEBUG_MODE:
            print(f"\tNumber of patches inside XTF: {self.patch_counter}")


    def create_patches(self, swaths: np.ndarray, 
                             waterfall: np.ndarray, 
                             xtf_file_name: str, 
                             start_idx: int, 
                             is_port: bool) -> None:
        """Creates sonar images from the current waterfall and creates the metadata information for each image.  

        Args:
            swaths (np.ndarray): UTM coordinates of each pixel inside the waterfall
            waterfall (np.ndarray): Waterfall image of the XTF file 
            xtf_file_name (str): XTF file path 
            start_idx (int): Starting row index of waterfall from which to extract sonar images 
            is_port (bool): Port (True) or starboard (False) 
        """

        assert swaths.shape[0] == waterfall.shape[0], "number of swaths same as waterfall height."
        assert swaths.shape[1] == waterfall.shape[1], "length of each swath same as waterfall width."

        if config.DEBUG_MODE:
            print(f"\tGenerating patches for {'port' if is_port else 'starboard'} side...")

        if config.PATCH_HEIGHT > waterfall.shape[0] or config.PATCH_WIDTH > waterfall.shape[1]:
            return 

        # Calculate the row and column indices of sonar images to extract from waterfall
        h, w = swaths.shape[:2]
        row_idxs = [r for r in range(0, h - config.PATCH_HEIGHT + 1, config.STRIDE_Y)]
        if h - row_idxs[-1] != config.PATCH_HEIGHT:
            row_idxs.append(h - config.PATCH_HEIGHT)
        col_idxs = [c for c in range(
            0, w - config.PATCH_WIDTH + 1, config.STRIDE_X)]
        if w - col_idxs[-1] != config.PATCH_WIDTH:
            col_idxs.append(w - config.PATCH_WIDTH)

        if config.DEBUG_MODE:
            first_row_patches = []
            last_row_patches = []

            if not hasattr(XTFPatchExtractor, 'plot_port_data') and not hasattr(XTFPatchExtractor, 'plot_stbd_data'):
                XTFPatchExtractor.plot_port_data = {}
                XTFPatchExtractor.plot_stbd_data = {}

        for row_idx in row_idxs:
            for col_idx in col_idxs:
                # Crop patch from waterfall
                patch_image = waterfall[row_idx:row_idx + config.PATCH_HEIGHT, col_idx:col_idx + config.PATCH_WIDTH]

                if config.DEBUG_MODE:
                    if row_idx == row_idxs[0] and len(first_row_patches) < 2:
                        first_row_patches.append(patch_image)
                    if row_idx == row_idxs[-1] and len(last_row_patches) < 2:
                        last_row_patches.append(patch_image)

                # Define patch file path
                max_patches = np.ceil(self.waterfall_shape[0] / config.STRIDE_Y) * np.ceil(self.waterfall_shape[1] / config.STRIDE_X)
                patch_file_name = str(self.patch_counter).zfill(len(str(max_patches))) + ".png"
                patch_file_path = os.path.join(self.patches_folder, patch_file_name) 

                # Add patch data to metadata rows
                metadata_row = [xtf_file_name, patch_file_path, 
                                "port" if is_port else "starboard", 
                                start_idx + row_idx, start_idx + row_idx + config.PATCH_HEIGHT, 
                                col_idx, col_idx + config.PATCH_WIDTH, config.EPSG_CODE,
                                swaths[row_idx,col_idx,0], swaths[row_idx,col_idx,1],
                                swaths[row_idx,col_idx+config.PATCH_WIDTH-1,0], swaths[row_idx,col_idx+config.PATCH_WIDTH-1,1],
                                swaths[row_idx+config.PATCH_HEIGHT-1,col_idx,0], swaths[row_idx+config.PATCH_HEIGHT-1,col_idx,1],
                                swaths[row_idx+config.PATCH_HEIGHT-1,col_idx+config.PATCH_WIDTH-1,0], swaths[row_idx+config.PATCH_HEIGHT-1,col_idx+config.PATCH_WIDTH-1,1]]

                # Increase counter
                self.patch_counter += 1

                # Append new sonar data
                self.sonar_data_rows.append(metadata_row)

        # Plot for debugging 
        if config.DEBUG_MODE:
            if is_port and xtf_file_name not in XTFPatchExtractor.plot_port_data.keys():
                self.plot_patches(first_row_patches, last_row_patches, is_port)
                XTFPatchExtractor.plot_port_data[xtf_file_name] = True

            if not is_port and xtf_file_name not in XTFPatchExtractor.plot_stbd_data.keys():
                self.plot_patches(first_row_patches, last_row_patches, is_port)
                XTFPatchExtractor.plot_stbd_data[xtf_file_name] = True    

    #######################################################################################################################################################
    #                                                                                                                                                     #
    #                                                       PLOTTING FUNCTIONS                                                                            #
    #                                                                                                                                                     #
    ####################################################################################################################################################### 


    def plot_xtf_summary(self, trajectory: np.ndarray) -> None:
        """Plot the trajectory of the side-scan-sonar for current XTF file on matplotlib

        Args:
            trajectory (np.ndarray): Trajectory of side-scan-sonar in UTM coordinates 
        """

        _, ax = plt.subplots(1,2,figsize=(10,5))

        ax[0].plot(trajectory[:,0], trajectory[:,1], color="blue", label="trajectory")
        ax[0].axis("equal")
        ax[0].set_title("Side-scan sonar trajectory")
        ax[0].set_xlabel("Easting [metres]")
        ax[0].set_ylabel("Northing [metres]")
        ax[0].scatter([trajectory[0,0]], [trajectory[0,1]], label="Start", c="red")
        ax[0].scatter([trajectory[-1,0]], [trajectory[-1,1]], label="End", c="green")
        ax[0].legend()

        data = np.asarray(self.sonar_data_rows)
        sonar_data_samples = {"port": (data[(data[:,2] == "port") , -8:].astype(np.float64), "red"), #& (data[:,3].astype(np.int64) < 500)
                            "starboard": (data[(data[:,2] == "starboard") , -8:].astype(np.float64), "green")}    #& (data[:,3].astype(np.int64) < 500)

        # Plot patches 
        for side, (data, colour) in sonar_data_samples.items():
            for i in range(data.shape[0]):
                topleft = data[i,:2]
                topright = data[i,2:4]
                bottomleft = data[i,4:6]
                bottomright = data[i,6:]
                if i == 0:
                    ax[1].plot([topleft[0],topright[0]],[topleft[1],topright[1]], color=colour, label=side)
                else:
                    ax[1].plot([topleft[0],topright[0]],[topleft[1],topright[1]], color=colour)
                ax[1].plot([topright[0],bottomright[0]],[topright[1],bottomright[1]], color=colour)
                ax[1].plot([bottomleft[0],bottomright[0]],[bottomleft[1],bottomright[1]], color=colour)
                ax[1].plot([bottomleft[0],topleft[0]],[bottomleft[1],topleft[1]], color=colour)


        ax[1].set_title(f"Patches of first 500 pings")
        ax[1].axis("equal")
        ax[1].set_xlabel("Easting [metres]")
        ax[1].set_ylabel("Northing [metres]")
        ax[1].legend()
        
        plt.show()


    def plot_waterfall(self, waterfall: np.ndarray, 
                             ping_idx: slice, 
                             port_idx: slice, 
                             stbd_idx: slice) -> None:
        """Plot section of waterfall on matplotlib

        Args:
            waterfall (np.ndarray): Waterfall image of the XTF file 
            ping_idx (slice): Row indices of waterfall
            port_idx (slice): Column indices of waterfall (port/left side)
            stbd_idx (slice): Column indices of waterfall (starboard/right side)
        """

        _, ax = plt.subplots(1,3, figsize=(10,5))

        waterfall_img = np.log10(waterfall[ping_idx,:] + 1e-6)
        waterfall_img.clip(0, np.max(waterfall_img), out=waterfall_img)
        ax[0].imshow(waterfall_img, cmap="gray")
        ax[0].set_title("Waterfall")

        port_img = np.log10(waterfall[ping_idx, port_idx] + 1e-6)
        port_img.clip(0, np.max(port_img), out=port_img)
        ax[1].imshow(port_img, cmap="gray")
        ax[1].set_title("Port")

        stbd_img = np.log10(waterfall[ping_idx, stbd_idx] + 1e-6)
        stbd_img.clip(0, np.max(stbd_img), out=stbd_img)
        ax[2].imshow(stbd_img, cmap="gray")
        ax[2].set_title("Starboard")

        plt.show()


    def plot_patches(self, first_row_patches: List[np.ndarray], 
                           last_row_patches: List[np.ndarray], 
                           is_port: bool) -> None:
        """Plot sonar images on matplotlib

        Args:
            first_row_patches (List[np.ndarray]): List of sonar images 
            last_row_patches (List[np.ndarray]): List of sonar images 
            is_port (bool): Port (True) or starboard (False) 
        """

        _, ax = plt.subplots(2,2,figsize=(7,7))
        for i,patch in enumerate(first_row_patches):
            patch = np.log10(patch + 1e-6)
            patch.clip(0, np.max(patch), out=patch)
            ax[0,i].imshow(patch, cmap="gray")
            ax[0,i].set_title(f"{'Port' if is_port else 'Starboard'} (1st row, {i+1}/{len(first_row_patches)})")

        for i,patch in enumerate(last_row_patches):
            patch = np.log10(patch + 1e-6)
            patch.clip(0, np.max(patch), out=patch)
            ax[1,i].imshow(patch, cmap="gray")
            ax[1,i].set_title(f"{'Port' if is_port else 'Starboard'} (nth row, {i+1}/{len(last_row_patches)})")