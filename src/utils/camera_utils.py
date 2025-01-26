#!/usr/bin/env python
"""
This script provides utility functions to work with cameras and images.
"""

# Import external libraries 
from sensor_msgs.msg import CompressedImage
from typing import List, Tuple, Optional
from cv_bridge import CvBridge
from math import sqrt
import numpy as np 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(cv_bridge=CvBridge())
def extract_image(msg: CompressedImage) -> np.ndarray:
    """Converts a CompressedImage ROS message to an OpenCV image 

    Args:
        msg (CompressedImage): input CompressedImage ROS message

    Returns:
        np.ndarray: converted OpenCV image 
    """

    return extract_image.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

def pixel_to_seabed(pixel_coord: Tuple[float, float], 
                    altitude: float, 
                    K: np.ndarray, 
                    T_world_to_camera: np.ndarray, 
                    D: Optional[List[float]] = None) -> Tuple[float, float]:
    """Projects a pixel coordinate from inside the image into world coordinates 

    Args:
        pixel_coord (Tuple[float, float]): Pixel coordinate 
        altitude (float): Altitude of camera from sea-bed in metres 
        K (np.ndarray): Intrinsic calibration matrix
        T_world_to_camera (np.ndarray): Extrinsic matrix _
        D (List[float] | None, optional): Distortion coefficients. Defaults to None.

    Returns:
        Tuple[float, float]: World coordinate (only x and y)
    """
  
    x_norm, y_norm, _ = np.matmul(np.linalg.inv(K), np.asarray(pixel_coord))
   
    if D is not None and len(D) == 5:
        r2 = x_norm**2 + y_norm**2
        radial_dist = 1.0 + D[0]*r2 + D[1]*r2**2 + D[4]*r2**3
        x_dist = x_norm * radial_dist + (2*D[2]*x_norm*y_norm + D[3]*(r2 + 2*x_norm**2))
        y_dist = y_norm * radial_dist + (2*D[3]*x_norm*y_norm + D[2]*(r2 + 2*y_norm**2))
    else:
        x_dist, y_dist = x_norm, y_norm

    norm = sqrt(altitude**2 + (x_dist * altitude)**2 + (y_dist * altitude)**2)
    x_cam = x_dist * norm
    y_cam = y_dist * norm
    z_cam = norm
    
    x_world, y_world, _, _ = np.matmul(T_world_to_camera, np.array([x_cam, y_cam, z_cam, 1.0]))
    return x_world, y_world

    