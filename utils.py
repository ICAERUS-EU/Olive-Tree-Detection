import math
import cv2
import numpy as np
import csv


def read_image_exif_data(image_path):
    import pyexiv2
    img = pyexiv2.Image(image_path)
    data = img.read_exif()
    img.close()
    return data


def read_image_xmp_data(image_path):
    fd = open(image_path, 'rb')
    d = fd.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end + 12]
    return xmp_str


def get_xmp_value(xmp, key):
    lines = xmp.split(b'\n')
    for line in lines:
        if line.find(key.encode('utf-8')) > 0:
            value = line.split(b'"')[1].decode('utf-8')
            return value


def get_dewarp_dict(xmp):
    data = get_xmp_value(xmp, 'DewarpData')
    data = data.split(';')
    day = data[0]
    data = data[1].split(',')
    dict = {
        'day': day,
        'fx': float(data[0]),
        'fy': float(data[1]),
        'cx': float(data[2]),
        'cy': float(data[3]),
        'dist': np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7]), float(data[8])])
    }
    return dict


def get_rgb_data(rgb_path):
    # Carica l'immagine
    rgb_img = cv2.imread(rgb_path)
    # Carica metadati XMP
    rgb_xmp = read_image_xmp_data(rgb_path)
    # Carica metedati EXIF
    rgb_exif = read_image_exif_data(rgb_path)
    
    # Crea il dizionario con i dati utili
    rgb_data = {
        'img': rgb_img,
        'centerX': float(get_xmp_value(rgb_xmp, 'CalibratedOpticalCenterX')),
        'centerY': float(get_xmp_value(rgb_xmp, 'CalibratedOpticalCenterY')),
        # 'vignetting': get_xmp_value(rgb_xmp, 'VignettingData'),
        'dewarp': get_dewarp_dict(rgb_xmp),
        'GimbalYawDegree' : float( get_xmp_value(rgb_xmp, 'GimbalYawDegree')),
        'GimbalRollDegree' : float( get_xmp_value(rgb_xmp, 'GimbalRollDegree')),
        'RelativeAltitude' : float( get_xmp_value(rgb_xmp, 'RelativeAltitude')),
        
        # 'hmatrix': get_xmp_value(rgb_xmp, 'CalibratedHMatrix'),
        # 'bps': rgb_exif['Exif.Image.BitsPerSample'],
        'focalLength': rgb_exif['Exif.Photo.FocalLength']
        # 'blackLevel': get_xmp_value(rgb_xmp, 'BlackLevel'),
        # 'sensorGain': get_xmp_value(rgb_xmp, 'SensorGain'),
        # 'exposureTime': get_xmp_value(rgb_xmp, 'ExposureTime'),
        # 'sensorGainAdjustment': get_xmp_value(rgb_xmp, 'SensorGainAdjustment'),
        # 'irradiance': get_xmp_value(rgb_xmp, 'Irradiance')
    }
    return rgb_data


def get_ms_data(ms_path):
    # Carica l'immagine
    ms_img = cv2.imread(ms_path)
    # Carica metadati XMP
    ms_xmp = read_image_xmp_data(ms_path)
    # Carica metedati EXIF
    ms_exif = read_image_exif_data(ms_path)
    # Crea il dizionario con i dati utili
    ms_data = {
        'img': ms_img,
        'centerX': float(get_xmp_value(ms_xmp, 'CalibratedOpticalCenterX')),
        'centerY': float(get_xmp_value(ms_xmp, 'CalibratedOpticalCenterY')),
        'vignetting': [float(x) for x in get_xmp_value(ms_xmp, 'VignettingData').split(', ')],
        'dewarp': get_dewarp_dict(ms_xmp),
        'dewarpHMatrix': np.asmatrix(np.array([float(x) for x in (get_xmp_value(ms_xmp, 'DewarpHMatrix').split(','))])).reshape(3,3),
        'hmatrix': np.asmatrix(np.array([float(x) for x in (get_xmp_value(ms_xmp, 'CalibratedHMatrix').split(','))])).reshape(3,3),
        'bps': int(ms_exif['Exif.Image.BitsPerSample']),
        'focalLength' : ms_exif['Exif.Photo.FocalLength'],
        'blackLevel': int(get_xmp_value(ms_xmp, 'BlackLevel')),
        'sensorGain': float(get_xmp_value(ms_xmp, 'SensorGain')),
        'exposureTime': int(get_xmp_value(ms_xmp, 'ExposureTime')),
        'sensorGainAdjustment': float(get_xmp_value(ms_xmp, 'SensorGainAdjustment')),
        'irradiance': float(get_xmp_value(ms_xmp, 'Irradiance'))
    }
    return ms_data


def check_first_value_consistency(boxes):
    if not boxes:
        return False
    first_value = boxes[0][0]
    for box in boxes:
        if box[0] != first_value:
            return False
    return True


def calculate_fov(fov_diagonal, aspect_ratio_width, aspect_ratio_height):
    diagonal = math.sqrt(aspect_ratio_width ** 2 + aspect_ratio_height ** 2)
    fov_horizontal = 2 * math.atan(math.tan(math.radians(fov_diagonal) / 2) * (aspect_ratio_width / diagonal))
    fov_horizontal = math.degrees(fov_horizontal)
    fov_vertical = 2 * math.atan(math.tan(math.radians(fov_diagonal) / 2) * (aspect_ratio_height / diagonal))
    fov_vertical = math.degrees(fov_vertical)
    return fov_horizontal, fov_vertical


def undistort_bounding_boxes(labels, fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height):
    """
    Applies undistortion to bounding box coordinates using camera calibration parameters.

    Parameters:
    labels (list of lists): Bounding boxes in the format [class_id, x_center, y_center, width, height].
    fx (float): Focal length along the x-axis.
    fy (float): Focal length along the y-axis.
    cx (float): Principal point along the x-axis.
    cy (float): Principal point along the y-axis.
    k1, k2, p1, p2, k3 (float): Distortion coefficients.
    width (int): Width of the image.
    height (int): Height of the image.

    Returns:
    list: Undistorted bounding boxes in the format [class_id, x_center, y_center, width, height].
    """
    undistort_boxes = []

    # Camera matrix
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients
    distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    for box in labels:
        class_id = box[0]
        x_center = box[1] * width
        y_center = box[2] * height
        box_width = box[3] * width
        box_height = box[4] * height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        # Distorted points
        distorted_points = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]], dtype=np.float32)
        distorted_points = np.expand_dims(distorted_points, axis=1)

        # Apply cv2.undistortPoints
        undistorted_points = cv2.undistortPoints(distorted_points, cameraMatrix, distCoeffs, P=cameraMatrix)
        undistorted_points = np.squeeze(undistorted_points)

        # Calculate the undistorted bounding box coordinates
        x_min = max(0, min(undistorted_points[:, 0]))
        y_min = max(0, min(undistorted_points[:, 1]))
        x_max = min(width, max(undistorted_points[:, 0]))
        y_max = min(height, max(undistorted_points[:, 1]))

        # Convert back to YOLO format
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        undistort_boxes.append([class_id, x_center, y_center, box_width, box_height])

    return undistort_boxes


def convert_filename(input_filename, str='.jpg'):
    parts = input_filename.split('_')
    new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}{str}"
    return new_filename


def read_mrk_file(file_path):
    """
    Reads the lines from a .mrk file.

    Parameters:
    file_path (str): The path to the .mrk file.

    Returns:
    list: A list of lines from the file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def process_mrk_line(line):
    """
    Processes a single line from a .mrk file to extract id, latitude, longitude, and ellipsoid height.

    Parameters:
    line (str): A single line from the .mrk file.

    Returns:
    dict: A dictionary with the extracted values.
    """
    parts = line.split('\t')
    id_val = parts[0]
    lat_val = [part for part in parts if ',Lat' in part][0].split(',')[0]
    lon_val = [part for part in parts if ',Lon' in part][0].split(',')[0]
    height_val = [part for part in parts if ',Ellh' in part][0].split(',')[0]
    return {'id': id_val, 'lat': lat_val, 'lon': lon_val, 'Ellh': height_val}


def write_to_csv(output_csv_path, data):
    """
    Writes the processed data to a CSV file.

    Parameters:
    output_csv_path (str): The path to the output CSV file.
    data (list): A list of dictionaries containing the processed data.
    """
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['id', 'lat', 'lon', 'Ellh']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def from_mrk_to_csv(file_path, output_csv_path):
    """
    Converts a .mrk file to a CSV file.

    Parameters:
    file_path (str): The path to the .mrk file.
    output_csv_path (str): The path to the output CSV file.
    """
    # Read the .mrk file
    lines = read_mrk_file(file_path)
    
    # Process each line
    data = [process_mrk_line(line) for line in lines]
    
    # Write the processed data to the CSV file
    write_to_csv(output_csv_path, data)
    
    print(f"Processed data saved to: {output_csv_path}")


