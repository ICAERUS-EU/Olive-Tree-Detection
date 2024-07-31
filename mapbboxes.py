import os
import pandas as pd
import cv2
import math
import numpy as np
from geometric_transformations import rotate_rectangle2d, reflection_point, x_axis_reflection, rotation_point2d
from utils import calculate_fov
from annotation_utils import write_yolo_labels

'''
df_boxes DataFrame structure
'xmin': The x-coordinate of the top-left corner of the bounding box.
'xmax': The x-coordinate of the bottom-right corner of the bounding box.
'ymin': The y-coordinate of the top-left corner of the bounding box.
'ymax': The y-coordinate of the bottom-right corner of the bounding box.
Given this, the CSV file might have a structure similar to the following:

xmin,xmax,ymin,ymax
752412.316577	752209.957278	4484150.0	4484099.0  (UTM coordinates)
752207.965472	752188.964679	4484132.0	4484105.0
...

'''


if __name__ == "__main__":

    # Constants
    file_paths_navs = ['./resources/navinfo/005_cimadimelfi_navinfo.csv']
    image_paths = ['./raw_data/DJI_202405251220_005_cimadimelfimisto']
    file_path_boxes_csv = './resources/cc_boxes/boxes_005_cimadimelfimisto.csv'
    output_dir = './raw_labels'

    # Main Processing Loop
    for file_path_nav, image_path in zip(file_paths_navs, image_paths):
        df_nav = pd.read_csv(file_path_nav)  # Read the navigation data
        
        df_boxes = pd.read_csv(file_path_boxes_csv)  # Read the boxes data

        align_rectangles = True  # Flag to determine whether to align rectangles
        
        positions = []  # List to store the positions
        for idx_nav, row_nav in df_nav.iterrows():
            image_name = row_nav['id_photo']  # Get the image name
            image_full_path = f'{image_path}/{image_name}'  # Create the full path for the image
            original_img = cv2.imread(image_full_path)  # Read the image
            height, width = original_img.shape[:2]  # Get the dimensions of the image
            distance_from_ground = row_nav["RelativeAltitude"]  # Get the distance from the ground
            FOV_RGB_horizontal, FOV_RGB_vertical = calculate_fov(fov_diagonal=84, aspect_ratio_width=width, aspect_ratio_height=height)

            Cx_geometric = width / 2  # Calculate the geometric center x
            Cy_geometric = height / 2  # Calculate the geometric center y
            
            angle_degree = row_nav["YawDegree"]  # Get the yaw angle in degrees
            reflection = int(row_nav["GimbalRollDegree"])  # Get the reflection angle
            position = row_nav["E"], row_nav["N"]  # Get the position (E, N)

            annotations = []
            for idx_box, row_box in df_boxes.iterrows():
                min_x = row_box['xmin'] - 1.25  # Adjust the xmin value
                max_x = row_box['xmax'] + 1.25  # Adjust the xmax value
                min_y = row_box['ymin'] - 1.25  # Adjust the ymin value
                max_y = row_box['ymax'] + 1.25  # Adjust the ymax value
                
                min_x, min_y = min_x - position[0], min_y - position[1]  # Adjust the coordinates relative to the position
                max_x, max_y = max_x - position[0], max_y - position[1]
            
                # Rotate the points
                bl = rotation_point2d(min_x, min_y, angle_degree=-angle_degree)
                tl = rotation_point2d(min_x, max_y, angle_degree=-angle_degree)
                br = rotation_point2d(max_x, min_y, angle_degree=-angle_degree)
                tr = rotation_point2d(max_x, max_y, angle_degree=-angle_degree)

                # Reflect the points if needed
                if reflection == 180:
                    bl = reflection_point(bl[0], bl[1], 0, 0)
                    tl = reflection_point(tl[0], tl[1], 0, 0)
                    br = reflection_point(br[0], br[1], 0, 0)
                    tr = reflection_point(tr[0], tr[1], 0, 0)
                
                # Calculate tangent points
                tan_bl = bl[0] / distance_from_ground, bl[1] / distance_from_ground
                tan_tl = tl[0] / distance_from_ground, tl[1] / distance_from_ground
                tan_br = br[0] / distance_from_ground, br[1] / distance_from_ground
                tan_tr = tr[0] / distance_from_ground, tr[1] / distance_from_ground
                
                # Calculate the tangent of half the FOV in radians
                tan_half_FOV_h = math.tan(math.radians(FOV_RGB_horizontal) / 2)
                tan_half_FOV_v = math.tan(math.radians(FOV_RGB_vertical) / 2)

                # Calculate ratios
                ratio_bl = tan_bl[0] / tan_half_FOV_h, tan_bl[1] / tan_half_FOV_v
                ratio_tl = tan_tl[0] / tan_half_FOV_h, tan_tl[1] / tan_half_FOV_v
                ratio_br = tan_br[0] / tan_half_FOV_h, tan_br[1] / tan_half_FOV_v
                ratio_tr = tan_tr[0] / tan_half_FOV_h, tan_tr[1] / tan_half_FOV_v

                # Reflect and scale points
                bl = ratio_bl[0] * width / 2 + Cx_geometric, ratio_bl[1] * height / 2 - Cy_geometric
                bl = x_axis_reflection(bl)

                tl = ratio_tl[0] * width / 2 + Cx_geometric, ratio_tl[1] * height / 2 - Cy_geometric
                tl = x_axis_reflection(tl)

                br = ratio_br[0] * width / 2 + Cx_geometric, ratio_br[1] * height / 2 - Cy_geometric
                br = x_axis_reflection(br)

                tr = ratio_tr[0] * width / 2 + Cx_geometric, ratio_tr[1] * height / 2 - Cy_geometric
                tr = x_axis_reflection(tr)

                # Convert points to integer coordinates
                bl_x_onpic = int(bl[0])
                bl_y_onpic = int(bl[1])

                tl_x_onpic = int(tl[0])
                tl_y_onpic = int(tl[1])

                br_x_onpic = int(br[0])
                br_y_onpic = int(br[1])

                tr_x_onpic = int(tr[0])
                tr_y_onpic = int(tr[1])

                # Check if the points are within the image bounds
                if not ((bl_x_onpic > width or bl_x_onpic < 0 or bl_y_onpic > height or bl_y_onpic < 0) and 
                        (br_x_onpic > width or br_x_onpic < 0 or br_y_onpic > height or br_y_onpic < 0) and 
                        (tl_x_onpic > width or tl_x_onpic < 0 or tl_y_onpic > height or tl_y_onpic < 0) and 
                        (tr_x_onpic > width or tr_x_onpic < 0 or tr_y_onpic < 0)): 
                    
                    x_minrect = min(bl_x_onpic, tl_x_onpic, br_x_onpic, tr_x_onpic)
                    x_maxrect = max(bl_x_onpic, tl_x_onpic, br_x_onpic, tr_x_onpic)
                    y_minrect = min(bl_y_onpic, tl_y_onpic, br_y_onpic, tr_y_onpic)
                    y_maxrect = max(bl_y_onpic, tl_y_onpic, br_y_onpic, tr_y_onpic)
                    
                    if align_rectangles:
                        verts = np.array([[bl_x_onpic, bl_y_onpic], [tl_x_onpic, tl_y_onpic], [tr_x_onpic, tr_y_onpic], [br_x_onpic, br_y_onpic]], np.int32)
                        verts = rotate_rectangle2d(verts, angle_degree, reflection=reflection)

                        x_minrect = min(verts[0][0], verts[1][0], verts[2][0], verts[3][0])
                        x_maxrect = max(verts[0][0], verts[1][0], verts[2][0], verts[3][0])
                        y_minrect = min(verts[0][1], verts[1][1], verts[2][1], verts[3][1])
                        y_maxrect = max(verts[0][1], verts[1][1], verts[2][1], verts[3][1])
                    
                    # Normalize coordinates to YOLO format
                    x_center = (x_minrect + x_maxrect) / 2 / width
                    y_center = (y_minrect + y_maxrect) / 2 / height
                    w = (x_maxrect - x_minrect) / width
                    h = (y_maxrect - y_minrect) / height

                    annotations.append((0, x_center, y_center, w, h))  # Assuming '0' as class id for 'Tree'

            # Save annotations to a text file in YOLO format
            yolo_filename = os.path.splitext(image_name)[0] + '.txt'
            write_yolo_labels(output_dir, yolo_filename, annotations)
