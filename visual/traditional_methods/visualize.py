import matplotlib.pyplot as plt
import cv2


def visualize_images(images, titles=None, cols=3, figsize=(15, 5)):
    """
    Visualize one or more images in a grid layout.

    Args:
        images (list or np.array): A list of images (grayscale or RGB).
        titles (list of str, optional): Titles for each image. Default is None.
        cols (int): Number of columns in the grid layout. Default is 3.
        figsize (tuple): Size of the entire figure. Default is (15, 5).
    """
    # Ensure `images` is a list for uniform processing
    if not isinstance(images, list):
        images = [images]

    # Number of images
    num_images = len(images)

    # Create grid layout
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap="gray")
        elif len(img.shape) == 3:  # RGB image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image shape: expected 2D or 3D arrays.")

        # Add title if provided
        if titles and i < len(titles):
            plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()
    plt.show()
