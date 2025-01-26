#!/usr/bin/env python
"""
This script provides utility functions to work with rotation and transformation matrices. 
"""

# Import external libraries 
import numpy as np 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"


def convert(x):
    if hasattr(x, "tolist"):  
        return x.tolist()
    raise TypeError(x)


def construct_T(tx: float, 
                ty: float, 
                tz: float, 
                roll_rad: float, 
                pitch_rad: float, 
                yaw_rad: float) -> np.ndarray:
    """Construct transformation matrix from translations and rotation angles (in radians)

    Args:
        tx (float): translation in x
        ty (float): translation in y
        tz (float): translation in z
        roll_rad (float): rotation in x in radians
        pitch_rad (float): rotation in y in radians
        yaw_rad (float): rotation in z in radians

    Returns:
        np.ndarray: Transformation matrix
    """

    r_rad = np.array([roll_rad, pitch_rad, yaw_rad])
    t = np.array([tx, ty, tz])
    Rz = np.array([[np.cos(r_rad[2]), -np.sin(r_rad[2]), 0], 
                    [np.sin(r_rad[2]), np.cos(r_rad[2]), 0],
                    [0,0,1]])
    Ry = np.array([[np.cos(r_rad[1]), 0, np.sin(r_rad[1])],
                    [0, 1, 0], 
                    [-np.sin(r_rad[1]), 0, np.cos(r_rad[1])]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(r_rad[0]), -np.sin(r_rad[0])], 
                    [0, np.sin(r_rad[0]), np.cos(r_rad[0])]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T


def construct_R(roll_rad: float, 
                pitch_rad: float, 
                yaw_rad: float) -> np.ndarray:
    """Construct rotation matrix from rotation angles (in radians)

    Args:
        roll_rad (float): rotation in x in radians
        pitch_rad (float): rotation in y in radians
        yaw_rad (float): rotation in z in radians

    Returns:
        np.ndarray: Rotation matrix
    """
    

    r_rad = np.array([roll_rad, pitch_rad, yaw_rad])
    Rz = np.array([[np.cos(r_rad[2]), -np.sin(r_rad[2]), 0], 
                    [np.sin(r_rad[2]), np.cos(r_rad[2]), 0],
                    [0,0,1]])
    Ry = np.array([[np.cos(r_rad[1]), 0, np.sin(r_rad[1])],
                    [0, 1, 0], 
                    [-np.sin(r_rad[1]), 0, np.cos(r_rad[1])]])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(r_rad[0]), -np.sin(r_rad[0])], 
                    [0, np.sin(r_rad[0]), np.cos(r_rad[0])]])
    return Rz @ Ry @ Rx

def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """Extracts euler angles (in radians) from rotation matrix

    Args:
        R (np.ndarray): rotation matrix

    Returns:
        np.ndarray: euler angles (in radians)
    """

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

