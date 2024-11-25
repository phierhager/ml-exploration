import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import filters


def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)


def apply_prewitt(image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(image, -1, kernel_x).astype(np.float32)
    prewitt_y = cv2.filter2D(image, -1, kernel_y).astype(np.float32)
    return cv2.magnitude(prewitt_x, prewitt_y)


def apply_canny(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)


def harris_corners(image, block_size=2, ksize=3, k=0.04):
    return cv2.cornerHarris(np.float32(image), block_size, ksize, k)


def shi_tomasi_corners(image, max_corners=50, quality_level=0.01, min_distance=10):
    # Detect corners
    output_image = image.copy()
    corners = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)

    # If corners are detected, overlay them on the image
    if corners is not None:
        corners = np.int32(corners)  # Convert corner coordinates to integer
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(
                output_image, (x, y), 5, (0, 255, 0), -1
            )  # Draw green circles on the corners
        return output_image

    # If no corners are detected, return the original grayscale image
    return output_image
