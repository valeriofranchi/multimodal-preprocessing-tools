# Multi-modal Pre-processing Tools

# 1. Description

**Author**: Valerio Franchi

This repository contains several pre-processing tools for multi-modal datasets containing navigation data and optical images from ROS Bags and side-scan-sonar (SSS) data from XTF files. 

These include the following:
- Debugging tools to visualise the raw and processed data inside the ROS bags and XTF files
- Camera and sonar image extraction 
- Navigation and camera metadata matching via timestamp matching
- Camera and sonar image matching via geographical constraints
- Multi-modal dataset construction from camera and sonar images  

# 2. Prerequisites 

- ROS1/ROS2 
- pyxtf 
- shapely 
- numpy
- geopandas
- matplotlib
- pandas 
- rosbags 

The repository contains a **requirements.txt** file that can be used to automatically install the required libraries to run the code. 

# 3. Installation

Clone the repository to a local folder:

```
git clone `https://github.com/valeriofranchi/multimodal-preprocessing-tools`
```

To install the required libraries there are two options:

1. Install them directly:

```
cd multimodal-preprocessing-tools
pip install -r requirements.txt
```

2. Set up a virtual environment and then install them (to avoid conflicts with other local projects containing other library versions):


If you already possess a virtual environment library, ignore the next section. This is an example with *virtualenv* but there are multiple other virtual environment libraries you can use. 


Install *virtualenv* and create a virtual environment:
```
pip install virtualenv
python<version> -m venv <virtual-environment-name>
```

Enter your virtual environment and install the required libraries:
```
source <virtual-environment-name>/bin/activate
pip install -r requirements.txt
```

# 4. Usage 

In order to use the library tools, there are two main steps:
1. Modify the parameters inside the configuration file to your liking
2. Run the main python script 

## Configuration File 

The configuration file named *config.py* contains the following optional parameters:

### Sonar parameters

- **PATCH_HEIGHT**: height of sonar image to be extracted 
- **PATCH_WIDTH**: width of sonar image to be extracted
- **STRIDE_X**: step size in horizontal direction by which to move across the input waterfall image to extract the sonar image 
- **STRIDE_Y** : step size in vertical direction by which to move across the input waterfall image to extract the sonar image  

### XTF parameters 

- **EPSG_CODE**: unique identifier for the UTM zone traversed during the data collection process 
- **SONAR_RPY_RAD**: boolean variable describing the unit of the angles inside the XTF files (radians or degrees)
- **XTF_UTC_TIME**: time zone difference from UTC+0 of the collected data inside the XTF files 

### Image parameters 

- **MAX_IMAGE_HEIGHT**: maximum altitude for image extraction
- **PERCENT_IMAGE_OVERLAP**: minimum overlap between sonar and camera image projections on sea-bed to be considered a match
- **SKIP_FRAMES**: ratio of number of images per topic skipped to number of images extracted during ROS bag processing  

### File locations

- **XTF_FOLDER_PATH**: path to input XTF files on disk  
- **ROSBAG_FOLDER_PATH**: path to input ROS bags on disk
- **PROCESS_FOLDER_PATH**: output path on disk where the pre-processed XTF files and ROS bags will be saved  
- **DATASET_FOLDER_PATH**: output path on disk where the dataset of sonar and camera image correspondences will be saved  

### ROS Parameters

- **NAVIGATION_TOPIC**: ROS navigation topic publishing [cola2_msgs/NavSts](https://bitbucket.org/iquarobotics/cola2_msgs/src/master/msg/) messages 
- **CAMERA_TOPICS**: list of ROS camera image topics publishing [sensor_msgs/CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) messages
- **CAMERA_INFO_TOPICS**: list of ROS camera info topics publishing [sensor_msgs/CameraInfo](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html) messages
- **USING_ROS2**: boolean variable describing the ROS version used to collect the data inside the bags (ROS2 or ROS1) 
- **ROSBAG_UTC_TIME**: time zone difference from UTC+0 of the collected data inside the ROS bags 

### Tool Options 

- **EXTRACT_PATCHES**: extracts sonar metadata from XTF file and saves it to disk
- **EXTRACT_NAVIGATION**: extracts navigation data from ROS bag and saves it to disk  
- **EXTRACT_IMAGES**: extracts camera metadata from ROS bag and saves it to disk
- **MATCH_IMAGES_TO_NAVIGATION**: matches camera and navigation data and updates the camera metadata (on disk) 
- **MATCH_IMAGES_TO_PATCHES**: matches sonar and camera data and saves metadata of correspondences 
- **RUN_ALL**: run all of the above
- **DEBUG_MODE**: shows visualisation graphics during execution
- **WAIT_FOR_USER_INPUT**: waits for user input during execution before continuing 
- **SAVE_PATCHES**: saves sonar images to disk during sonar metadata extraction
- **SAVE_IMAGES**: saves camera images to disk during camera metadata extraction
- **BUILD_DATASET**: saves sonar and camera images during sonar and camera matching  

## Run main.py

Once the parameters are set inside *config.py*, run the main python file:

```
python<version> main.py
```


# 5. Licence 

The source code is released under [GPLv3](https://www.gnu.org/licenses/) licence. 

For any technical issues, please contact Valerio Franchi <valerio.franchi@udg.edu>. 

<!-- If you use this repository in an academic work, please cite:

```


``` -->