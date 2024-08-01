<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">Olive Tree Detection🦚</h3>
    
   <p align="center">
    Training and validation of a detection model for olive trees
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/wiki"><strong>Explore the wiki »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/issues">Report Bug</a>
    -
    <a href="https://github.com/icaerus-eu/repo-title/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/repo-title/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/repo-title?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/repo-titlee?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/repo-title?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/repo-title) ![License](https://img.shields.io/github/license/icaerus-eu/repo-title) 


## Table Of Contents
- [Table Of Contents](#table-of-contents)
- [Summary](#summary)
- [Installation](#installation)
- [Documentation](#documentation)
  - [annotation\_utils.py](#annotation_utilspy)
  - [augmentation\_yoloboxes.py](#augmentation_yoloboxespy)
  - [geometric\_transformations.py](#geometric_transformationspy)
  - [process\_mrk\_to\_navinfo.py](#process_mrk_to_navinfopy)
  - [mapbboxes.py](#mapbboxespy)
  - [utils.py](#utilspy)
  - [train\_detection.py](#train_detectionpy)
  - [validation\_detection.py](#validation_detectionpy)
- [Acknowledgements](#acknowledgements)

## Summary
The provided scripts were instrumental in creating annotations and preparing the dataset needed to train a ```YOLOv9e``` model for detecting olive tree canopies from aerial images captured by a [DJI Mavic 3M drone](https://ag.dji.com/it/mavic-3-m/specs). The workflow involved managing bounding box annotations, augmenting images, and applying geometric transformations to enhance the model's robustness. Additionally, the scripts facilitated the training and validation processes, resulting in a model capable of accurately detecting olive trees from an aerial perspective

## Installation
Install python 3.10 and requred libraries listed in ```requirements.txt```.
The raw drone data has been divided into two repositories available at these links:
  - raw drone sensing data - 1: https://zenodo.org/records/12943015
  - raw drone sensing data - 2:  https://zenodo.org/records/12924461 

The dataset used for detection is available at https://zenodo.org/records/13121962.
The trained models are available at https://zenodo.org/records/13149720.


## Documentation


### annotation_utils.py
The file ```annotation_utils.py``` contains useful functions for managing annotations in various formats, particularly for datasets used in object detection models. The main functions include reading and converting annotations between the format used by ```albumentations``` and that used by ```YOLO```.

Reading annotations in albumentations format: 
The ```read_albumentations_annotations``` function reads an annotation file and returns a list of annotations. Each annotation is represented as a list containing a class ID and the bounding box coordinates ```(x_min, y_min, x_max, y_max)```.

Converting annotations from albumentations to YOLO format: 
The ```albumentations_to_yolo``` function converts annotations from the albumentations format to the YOLO format. YOLO annotations are represented by a class ID, the bounding box center coordinates, and the bounding box dimensions (width and height), all normalized with respect to the image dimensions.

Reading YOLO labels: 
The ```read_yolo_labels``` function reads a YOLO label file and returns a list of annotations, each represented with a class ID, the bounding box center coordinates, and its dimensions.


### augmentation_yoloboxes.py
The ```augmentation_yoloboxes.py``` file focuses on applying geometric transformations to images and their corresponding bounding box annotations in YOLO format. It utilizes the albumentations library to define and apply these transformations.

Imports and Setup:
The file begins by importing several libraries, including ```albumentations``` for transformations, ```cv2``` for image processing, and functions from the ```annotation_utils.py``` file for managing annotations. The paths for input and output directories for images and labels are also defined.

Creation of Output Directories: 
Before applying the transformations, the code checks if the output directories exist and creates them if they do not, ensuring that the results can be saved correctly.

Definition of Transformations: 
A list of geometric transformations is created to be applied to the images and annotations. These include operations like horizontal and vertical flips, rotations at various angles, central cropping, and resizing. Each transformation is defined using ```A.Compose``` from albumentations and includes the necessary parameters to manage the bounding boxes consistently. The transformed images are cropped and resized to a final size of ```640x960``` pixels.


### geometric_transformations.py
The file ```geometric_transformations.py``` contains a series of mathematical functions for performing geometric transformations on points and calculating distances relative to geometric figures. These functions can be useful in various contexts, such as image manipulation or computational geometry.

Reflection of a point relative to another point: 
The ```reflection_point``` function calculates the reflection of a point relative to a given point, returning the coordinates of the reflected point.

Reflection of a point relative to the x-axis: 
The ```x_axis_reflection``` function reflects a point relative to the x-axis, inverting its y-coordinate.

Rotation of a point around the origin: 
The ```rotation_point2d``` function rotates a point around the origin by a given angle in degrees, returning the new coordinates of the rotated point.

Distance from a rectangle: 
The ```distance_from_rectangle``` function calculates the squared distance of a point from the nearest edge of a rectangle centered at a given point with a specified width and height.


### process_mrk_to_navinfo.py
The ```process_mrk_to_navinfo.py``` file is focused on processing data from ```.MRK``` files generated by DJI drones, which contain temporal and navigational information. The file processes this data and transforms it into a more useful format for further analysis, such as conversion to CSV and enhancement with geoid height information.

Imports and Setup:
The file imports several libraries, including ```pandas``` for data manipulation, ```requests``` and ```BeautifulSoup``` for extracting web data, and other libraries useful for image processing and coordinate conversion.

File and Directory Paths:
At the beginning of the file, the paths for input and output files are defined, including the ```.MRK``` files, resulting CSVs, and image directories.

Retrieving Geoid Height: 
The ```get_geoid_height_egm2008``` function retrieves the geoid height using the EGM2008 model for a given latitude and longitude by making a request to an online service and parsing the HTML response to extract the necessary data.

Calculating Orthometric Height: 
The ```calculate_orthometric_height``` function calculates the orthometric height for each row in a DataFrame using the geoid height and other altitude information.

Converting MRK to CSV: 
A function is used to convert ```.MRK``` data into CSV format, facilitating further processing and analysis.


### mapbboxes.py
The ```mapbboxes.py``` file is dedicated to processing data related to bounding boxes and images, using navigational information to map the coordinates of the bounding boxes onto specific images. This is useful in contexts such as photogrammetry and computer vision, where it is necessary to align and transform bounding boxes based on navigation data and the orientation of the images.

Imports and Setup: 
The file begins by importing several libraries, including ```pandas``` for data manipulation, ```cv2``` for image processing, and custom functions from other files like ```geometric_transformations.py``` and ```annotation_utils.py```.

Bounding Box Data Structure: 
The file includes a description of the structure of the ```df_boxes``` DataFrame, which contains UTM (Universal Transverse Mercator) coordinates of the bounding box vertices. This data is read from a CSV file.

Main Processing Loop: 
In the main function, the code iterates through the navigation data and associated images, reading the bounding box coordinates and the images themselves. During processing, the bounding boxes can be aligned and transformed based on the orientation and position information provided by the navigation data.

Alignment and Transformation of Bounding Boxes: 
The code includes a flag to determine whether the bounding boxes should be aligned. It uses geometric transformation functions to adjust the bounding boxes based on the image's orientation and the distance from the ground.

Projection of Points from Boxes onto the RGB Photo: 
For the bounding boxes identified with the Label Connected Component, the points from the 3D point cloud are projected onto the RGB photo. To identify the portion of the point cloud contained in the RGB photo, the position data (latitude, longitude, altitude) of the drone at the time the photo was taken is used, along with the EXIF metadata contained in the photo, including ```GimbalYawAngle```, ```GimbalRollDegree```, and FOV. Using the drone's position data, the camera's location in three-dimensional space can be approximated. The gimbal metadata (yaw and roll) helps to establish the exact direction in which the camera was oriented. With the camera's orientation and FOV, it is possible to project a frustum (visual cone) from the photo. This frustum represents the three-dimensional area captured by the camera. The point cloud is then filtered to include only the points that fall within this frustum. These points represent the portion of the point cloud that is visible in the RGB photo.


### utils.py
The ```utils.py``` file contains a set of general utility functions for reading and manipulating images and their metadata, such as EXIF and XMP data, which are commonly used in images captured with modern cameras, especially in drones.

Reading EXIF Data: 
The ```read_image_exif_data``` function uses the ```pyexiv2``` library to read EXIF data from an image. EXIF data contains information such as the date, time of capture, camera settings, and more.

Reading XMP Data: 
The ```read_image_xmp_data``` function extracts XMP metadata from an image file. XMP metadata can contain additional information compared to EXIF, such as image correction data and camera parameters.

Extracting Specific Values from XMP Data: 
The ```get_xmp_value``` function searches for and returns a specific value from the XMP metadata, given a specific key.

Retrieving Dewarp Data: 
The ```get_dewarp_dict``` function extracts and organizes dewarp (distortion correction) data from XMP metadata, returning a dictionary with information such as optical calibration parameters.

Loading RGB Data and Associated Metadata: 
The ```get_rgb_data``` function loads an RGB image and its associated EXIF and XMP metadata, creating a dictionary that contains the image itself and useful information for further processing, such as the calibrated optical center and the camera's tilt angle


### train_detection.py
The ```train_detection.py``` file is a simple and intuitive script designed for training an object detection model using the Ultralytics library, based on the YOLO architecture. This script allows you to start or resume training a YOLO model on a specific dataset, making the process straightforward.

The script begins by importing the Ultralytics library, which provides the necessary functionalities for training and using YOLO models. Next, some fundamental configurations are defined: the version of the YOLO model to be used, in this case, ```yolov9e.pt```, the path to the training dataset specified in a ```data.yaml``` file, and the folder where the training results will be saved.

The start of the training is controlled by a flag called start. If this flag is set to True, the model begins training using the specified parameters, such as image size, number of epochs (80 epochs in total), batch size, and the possible use of data augmentation techniques. This allows the model to learn and optimize its performance on the provided dataset.

If the flag start is set to False, the script loads a previously trained YOLO model and prepares it to resume training from where it was left off, ensuring continuity in the model's optimization process


### validation_detection.py
The file ```validation_detection.py``` is a script designed to validate a pre-trained YOLO object detection model using the Ultralytics library. This script allows for the evaluation of the model's performance on a test dataset, generating various validation metrics that are then saved for subsequent analysis.

The script begins by importing the necessary libraries: ```Ultralytics``` to handle the YOLO model, ```pandas``` to manipulate data in DataFrame format and save it as CSV files, and ```json``` to save the validation results in JSON format. Next, the path of the trained model is specified, along with the folder where the validation results will be saved.

Once the YOLO model is successfully loaded, the script proceeds with the validation phase using the test dataset. During this process, plots illustrating the model's performance are generated and saved along with the results. The main metrics, such as precision, recall, mAP50, and mAP50-95, are extracted from the validation results.

Finally, these results are stored in a JSON file, making it easier to access and analyze the model's performance. The script also saves CSV files containing the validation curves, allowing for more detailed analysis.


## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>
