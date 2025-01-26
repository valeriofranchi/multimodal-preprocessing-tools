#!/usr/bin/env python
"""
This script provides utility functions to work with side-scan-sonar data.
"""

# Import external libraries 
from typing import List, Any, Tuple 
from pyproj import Proj, CRS
import numpy as np 
import pyxtf 
import os 

# Import internal scripts
from src import config 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"


def load_xtf(xtf_file_path: str) -> List[Any]:
    """Extract XTF ping data from XTF file 

    Args:
        xtf_file_path (str): XTF file path 

    Raises:
        FileNotFoundError: file not found on disk 
        TypeError: file not correct file type 

    Returns:
        List[Any]: XTF ping data 
    """

    if not os.path.exists(xtf_file_path):
        raise FileNotFoundError("Invalid Path", xtf_file_path)

    if not xtf_file_path.endswith('.xtf'):
        raise TypeError("Invalid File", xtf_file_path)
    
    (_, packet) = pyxtf.xtf_read(xtf_file_path)
    xtf_pings = packet[pyxtf.XTFHeaderType.sonar]
    
    print("XTF File loaded!")
    return xtf_pings


def calculate_blind_zone_indices(xtf_pings: List[Any]) -> Tuple[slice, slice]:
    """Calculates indices of waterfall's non-blind zone

    Args:
        xtf_pings (List[Any]): Input XTF ping data 

    Returns:
        Tuple[slice, slice]: Waterfall column indices corresponding to pixels that do not fall in blind zone 
    """

    num_samples = xtf_pings[0].ping_chan_headers[0].NumSamples * 2
    slant_res = xtf_pings[0].ping_chan_headers[0].SlantRange * 2 / num_samples
    altitude = np.max([ping.SensorPrimaryAltitude for ping in xtf_pings])

    num_bins_blind = int(round(altitude / slant_res))            
    num_bins_ground = int(num_samples / 2 - num_bins_blind)   

    port_idx = slice(0, num_bins_ground)
    stbd_idx = slice(num_bins_ground + 2 * num_bins_blind, -1)

    if config.DEBUG_MODE:
        print(f"\tNumber of blind bins: {num_bins_blind*2}")

    return port_idx, stbd_idx

def calculate_swath_positions(xtf_pings: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates geographical locations of the pixels of the waterfall, and the trajectory of the side-scan-sonar

    Args:
        xtf_pings (List[Any]): Input XTF ping data 

    Returns:
        Tuple[np.ndarray, np.ndarray]: Geographical locations of the pixels of the waterfall, and the trajectory of the side-scan-sonar
    """

    lonlat_to_EN = Proj(CRS.from_epsg(config.EPSG_CODE), preserve_units=False)
    ping_info = xtf_pings[0].ping_chan_headers[0]

    # Fetch data dimensions
    num_pings = len(xtf_pings)
    num_samples = ping_info.NumSamples * 2

    if config.DEBUG_MODE:
        print(f"Number of pings: {num_pings}\nNumber of samples: {num_samples}")

    # Compute swath resolution
    slant_range = ping_info.SlantRange         
    slant_res = slant_range * 2 / num_samples  

    if config.DEBUG_MODE:
            print(f"Slant range: {slant_range} [metres]\nSlant resolution: {slant_res} [metres/bin]")

    # Fetch navigation parameters
    longitude, latitude, altitude, roll, pitch, yaw = zip(*[(ping.SensorXcoordinate, ping.SensorYcoordinate,
                                                                ping.SensorPrimaryAltitude, ping.SensorRoll,
                                                                ping.SensorPitch, ping.SensorHeading)
                                                                for ping in xtf_pings])

    east, north = lonlat_to_EN(longitude, latitude)
    altitude = np.asarray(altitude).reshape(num_pings, 1)
    if not config.SONAR_RPY_RAD:
        roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)

    bin_central = (num_samples - 1) / 2
    bins = np.arange(num_samples).reshape(1, num_samples)          
    bins_from_center = bins - bin_central                      

    # Number of bins corresponding to the Blind Zone per side
    n_bins_blind = np.round(altitude / slant_res)    

    # Number of bins corresponding to Ground Range per side
    n_bins_ground = (num_samples / 2 - n_bins_blind)            

    # Bins inside the blind zone
    blind_idx = (n_bins_ground <= bins) & \
                (bins < n_bins_ground + 2 * n_bins_blind)      

    # Increments along the x-axis (swath width)
    X = np.zeros((num_pings, num_samples))
    np.sqrt(np.square(slant_res * bins_from_center) -
            np.square(altitude).reshape(num_pings, 1),
            where=~blind_idx, out=X)
    X *= np.sign(bins_from_center)                         

    # Ping coordinates (swath center)
    T = np.vstack((east, north)).T      

    # Rotation of x-axis (swath) about the z-axis (heading)
    R = np.vstack((np.cos(yaw), -np.sin(yaw))).T               

    X = np.expand_dims(X, axis=2)                          
    T = np.expand_dims(T, axis=1)                          
    R = np.expand_dims(R, axis=1)                          

    # Compute the transformation
    swaths = T + R * X       

    print("Swath positions calculated!")  
    if config.DEBUG_MODE:
        print(f"Swath positions shape: {swaths.shape}")   
                            
    trajectory = np.vstack([east, north]).T
    return swaths, trajectory

def calculate_waterfall(xtf_pings: List[Any], 
                        channels: List[int] = [0,1]) -> np.ndarray:
    """Calculates waterfall from XTF ping data 

    Args:
        xtf_pings (List[Any]): Input XTF ping data 
        channels (List[int], optional): Channel indices from the XTF file. Defaults to [0,1].

    Returns:
        np.ndarray: Waterfall image 
    """

    num_channels = len(xtf_pings[0].data)
    assert len(channels) ==2 and channels[0] < num_channels and channels[1] < num_channels
    
    # Load all the intensity information 
    sonar_chans = [np.vstack([ping.data[i] for ping in xtf_pings]) for i in channels]

    # Stack port and starboard data side by side 
    sonar_image = np.hstack( (sonar_chans[channels[0]], sonar_chans[channels[1]]) )

    if config.DEBUG_MODE:
        print(f"Waterfall shape: {sonar_image.shape}")

    waterfall_img = np.log10(sonar_image + 1e-6)
    waterfall_img.clip(0, np.max(waterfall_img), out=waterfall_img)
    waterfall_img = (waterfall_img - waterfall_img.min()) / waterfall_img.max() * 255.0
    return waterfall_img
