import albumentations as A
import cv2
import os
from annotation_utils import read_yolo_labels, write_yolo_labels, albumentations_to_yolo, yolo_to_albumentations, clip_bboxes


def filter_boxes_byratio(bboxes):
    """
    Filters bounding boxes based on their aspect ratio and area.
    
    Parameters:
    bboxes (list of lists): List of bounding boxes, where each bounding box is represented as 
                            [class_id, x_center, y_center, width, height].
    
    Returns:
    filtered_bboxes (list of lists): List of bounding boxes that meet the filtering criteria.
    removed_count (int): Number of bounding boxes that were removed during filtering.
    """
    filtered_bboxes = []
    removed_count = 0
    for box in bboxes:
        class_id,x_center, y_center, width, height = box
        aspect_ratio = width / height
        area = width * height
        if 0.33 <= aspect_ratio <= 3:
            if not (0.5 <= aspect_ratio <= 2) and area <= 0.01:
                removed_count += 1
            else:
                filtered_bboxes.append([class_id,x_center,y_center,width,height])
        else:
            removed_count += 1

    return filtered_bboxes,removed_count


def augmentations_rgb(input_images_dir, input_labels_dir, filter_boxes_by_ratio=True):
    """
    Applies augmentations to images and their corresponding labels in YOLO format for training an object detection network with RGB photos.

    Parameters:
    use_color_transformations (bool): Flag to determine whether to use color transformations.
    """
    transformations = [
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.Affine(rotate=[-90, 90], p=1, mode=cv2.BORDER_CONSTANT, cval=(0, 0, 0), fit_output=False),
             A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.Affine(rotate=[180, 180], p=1, mode=cv2.BORDER_CONSTANT, cval=(0, 0, 0), fit_output=False),
             A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.CenterCrop(height=512, width=768),
            A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.Rotate(limit=(15, 20), p=1.0),
            A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])),
        A.Compose([
            A.Rotate(limit=(-20, -15), p=1.0),
            A.Resize(640, 960)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))
    ]

    for image_name in os.listdir(input_images_dir):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_images_dir, image_name)
            label_path = os.path.join(input_labels_dir, os.path.splitext(image_name)[0] + '.txt')

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = read_yolo_labels(label_path)
            bboxes = yolo_to_albumentations(bboxes)
            bboxes = clip_bboxes(bboxes)
            category_ids = [bbox[0] for bbox in bboxes]
            bboxes = [bbox[1:] for bbox in bboxes]
            
            # Apply each transformation
            for i, transform in enumerate(transformations):
                augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                transformed_image = augmented['image']
                transformed_bboxes = augmented['bboxes']
                
                transformed_bboxes_yolo = albumentations_to_yolo(transformed_bboxes)
                    
                transformed_labels = [[category_ids[j]] + list(transformed_bboxes_yolo[j]) for j in range(len(transformed_bboxes_yolo))]
                
                if i==0:
                    transformed_image_name = f"{os.path.splitext(image_name)[0]}.jpg"
                else:
                    transformed_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
                transformed_image_path = os.path.join(output_images_dir, transformed_image_name)
                cv2.imwrite(transformed_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                transformed_label_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.txt"
                
                if filter_boxes_by_ratio:
                    transformed_labels,_ = filter_boxes_byratio(transformed_labels)

                write_yolo_labels(output_dir=output_labels_dir,filename=transformed_label_name,annotations= transformed_labels)


if __name__ == "__main__":

    input_labels_dir = './dataset/train_original/labels'
    input_images_dir = './dataset/train_original/images'

    output_images_dir = './dataset/train/images'
    output_labels_dir = './dataset/train/labels'

    # Create output folders
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    augmentations_rgb(input_images_dir,input_labels_dir)



