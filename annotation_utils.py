import os
import numpy as np


def read_albumentations_annotations(file_path):
    """
    Reads annotations from a file in 'albumentations' format.

    Parameters:
    file_path (str): The path to the annotation file.

    Returns:
    list: A list of annotations, each represented as [class_id, x_min, y_min, x_max, y_max].
    """
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_min = float(parts[1])
            y_min = float(parts[2])
            x_max = float(parts[3])
            y_max = float(parts[4])
            annotations.append([class_id, x_min, y_min, x_max, y_max])
    return annotations


def albumentations_to_yolo(annotations):
    """
    Converts annotations from 'albumentations' format to 'yolo' format.

    Parameters:
    annotations (list): A list of annotations in 'albumentations' format.
    img_width (int): The width of the image.
    img_height (int): The height of the image.

    Returns:
    list: A list of annotations in 'yolo' format, each represented as [ x_center, y_center, width, height].
    """
    yolo_annotations = []
    for annotation in annotations:
        
        x_min = annotation[0]
        y_min = annotation[1] 
        x_max = annotation[2] 
        y_max = annotation[3] 
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = (x_max - x_min) 
        height = (y_max - y_min) 
        
        yolo_annotations.append([ x_center, y_center, width, height])
    return yolo_annotations


def read_yolo_labels(label_path):
    """
    Reads YOLO labels from a file.

    Parameters:
    label_path (str): The path to the YOLO label file.

    Returns:
    list: A list of YOLO labels, each represented as [class_id, x_center, y_center, width, height].
    """
    with open(label_path, 'r') as file:
        labels = []
        for line in file:
            parts = line.strip().split()
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append([int(parts[0]), x_center, y_center, width, height])
    return labels


def clip_bboxes(bboxes):
    clipped_bboxes = []
    
    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        x_min = np.clip(x_min, 0.0, 1.0)
        y_min = np.clip(y_min, 0.0, 1.0)
        x_max = np.clip(x_max, 0.0, 1.0)
        y_max = np.clip(y_max, 0.0, 1.0)
        if x_min < x_max and y_min < y_max:
            clipped_bboxes.append([class_id,x_min, y_min, x_max, y_max])
   
    return clipped_bboxes


def yolo_to_albumentations(annotations):
    albumentations_annotations = []
    for annotation in annotations:
        class_id = annotation[0]
        x_center = annotation[1] 
        y_center = annotation[2] 
        width = annotation[3] 
        height = annotation[4] 
        
        x_min = x_center - width/2
        y_min = y_center - height/2
        x_max = x_center + width/2
        y_max = y_center + height/2
        
        albumentations_annotations.append([class_id, x_min, y_min, x_max, y_max])
    return albumentations_annotations


def write_yolo_labels(output_dir, filename, annotations):
    """
    Writes YOLO labels to a file.

    Parameters:
    output_dir (str): The directory where the file will be saved.
    filename (str): The name of the file.
    annotations (list): The list of YOLO annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, filename)
    with open(output_file_path, 'w') as file:
        for annotation in annotations:
            file.write(f"{annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n")


