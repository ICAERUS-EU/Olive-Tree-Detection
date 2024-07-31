import math
import numpy as np


def reflection_point(x, y, xr, yr):
    """
    Calculate the reflection of a point (x, y) with respect to a point (xr, yr).

    Parameters:
    x, y: Coordinates of the original point.
    xr, yr: Coordinates of the point with respect to which the reflection occurs.

    Returns:
    tuple: Coordinates (x', y') of the reflected point.
    """
    # Calculate the coordinates of the reflected point
    x_reflected = 2 * xr - x
    y_reflected = 2 * yr - y

    return (x_reflected, y_reflected)


def x_axis_reflection(point):
    """
    Reflect a point across the x-axis.

    Parameters:
    point (tuple): The original point (x, y).

    Returns:
    tuple: The reflected point (x, -y).
    """
    x, y = point

    return (x, -y)


def rotation_point2d(x, y, angle_degree):
    """
    Rotate a point around the origin by a given angle.

    Parameters:
    x, y: Coordinates of the original point.
    angle_degree (float): The rotation angle in degrees.

    Returns:
    tuple: The rotated point (x', y').
    """
    angle_radian = math.radians(-angle_degree)
    x_rotated = x * math.cos(angle_radian) - y * math.sin(angle_radian)
    y_rotated = x * math.sin(angle_radian) + y * math.cos(angle_radian)

    return x_rotated, y_rotated


def distance_from_rectangle(x, y, cx, cy, width, height):
    """
    Calculate the squared distance from a point (x, y) to the closest edge of a rectangle
    centered at (cx, cy) with the specified width and height.

    Parameters:
    x, y (float): Coordinates of the point.
    cx, cy (float): Coordinates of the rectangle's center.
    width, height (float): Width and height of the rectangle.

    Returns:
    float: The squared distance from the point to the closest edge of the rectangle.
    """
    # Calculate the boundaries of the rectangle
    left = cx - width / 2
    right = cx + width / 2
    top = cy - height / 2
    bottom = cy + height / 2

    # Check if the point is inside the rectangle
    if left <= x <= right and top <= y <= bottom:
        return 0.0

    # Calculate the horizontal and vertical distances to the nearest edge
    dx = max(left - x, 0, x - right)
    dy = max(top - y, 0, y - bottom)

    # Return the squared distance
    return dx**2 + dy**2


def rotate_rectangle2d(vertices, angle_deg, reflection):
    """
    Rotate a rectangle based on the given angle and handle reflection if needed.

    Parameters:
    vertices (numpy array): The vertices of the rectangle.
    angle_deg (float): The angle in degrees to rotate.
    reflection (float): The reflection angle.

    Returns:
    numpy array: The rotated (and possibly reflected) vertices.
    """
    # Calculate the center point of the rectangle
    center = np.mean(vertices, axis=0)

    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle_deg)

    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Translate the rectangle to the center point
    translated_vertices = vertices - center

    # Rotate the rectangle
    rotated_vertices = np.dot(translated_vertices, rotation_matrix.T) + center
    
    # Check for reflection condition
    if reflection == 180.0:
        # Invert the x-coordinates (reflection across the y-axis)
        rotated_vertices[:, 0] = 2 * center[0] - rotated_vertices[:, 0]

    return rotated_vertices
