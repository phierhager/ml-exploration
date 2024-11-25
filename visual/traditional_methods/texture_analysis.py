import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import filters
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt


def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    _, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
    )
    return hog_image


def compute_lbp(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp


def compute_glcm(
    image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
):
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )
    return glcm

def show_glcm(glcm):
    # Normalize the GLCM for visualization
    glcm_normalized = glcm / glcm.max()
    plt.imshow(glcm_normalized[:, :, 0], cmap="gray")  # Show the first angle's GLCM
    plt.title("GLCM Matrix")
    plt.colorbar()  # Optional: Add a color bar for reference
    plt.axis("off")
    plt.show()
