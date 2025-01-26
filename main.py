#!/usr/bin/env python
"""
Main file
"""

# Import internal scripts
from src import config
from src.extraction.patch_extractor import XTFPatchExtractor
from src.extraction.navigation_extractor import NavigationExtractor
from src.extraction.camera_extractor import CameraExtractor
from src.matching.camera_to_navigation_matcher import CameraToNavigationMatcher
from src.matching.camera_to_sonar_matcher import CameraToSonarMatcher 

__author__ = "Valerio Franchi"
__copyright__ = "Copyright 2025, CIRS, Universitat de Girona"
__credits__ = ["Valerio Franchi"]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Valerio Franchi"
__email__ = "valerio.franchi@udg.edu"
__status__ = "Prototype"


if __name__ == '__main__':

    # Print parameters and save them to disk 
    config.printParameters()
    config.saveParameters()

    # Run everything
    if config.RUN_ALL:
        extract_patches = XTFPatchExtractor()
        extract_nav = NavigationExtractor()
        extract_images = CameraExtractor()
        match_images_to_nav = CameraToNavigationMatcher()
        match_images_to_sonar = CameraToSonarMatcher()

    # Run only specific functions 
    else:
        # Extract patches 
        if config.EXTRACT_PATCHES:
            extract_patches = XTFPatchExtractor()

        # Extract navigation
        if config.EXTRACT_NAVIGATION:
            extract_nav = NavigationExtractor()

        # Extract camera images
        if config.EXTRACT_IMAGES:
            extract_images = CameraExtractor()

        # Match images to navigation data
        if config.MATCH_IMAGES_TO_NAVIGATION:
            match_images_to_nav = CameraToNavigationMatcher()

        # Match images to sonar data 
        if config.MATCH_IMAGES_TO_PATCHES:
            match_images_to_sonar = CameraToSonarMatcher()