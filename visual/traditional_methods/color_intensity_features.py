import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_color_histogram(image, bins=256):
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return hist


def compute_color_moments(image):
    mean = np.mean(image, axis=(0, 1))
    std_dev = np.std(image, axis=(0, 1))
    skewness = np.mean((image - mean) ** 3, axis=(0, 1)) / (std_dev**3)
    return mean, std_dev, skewness


def normalize_rgb(image):
    norm = np.sum(image, axis=-1, keepdims=True)
    normalized = image / (norm + 1e-8)
    return normalized


def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def visualize_color_histogram(hist, bins=256):
    plt.figure(figsize=(10, 5))
    plt.title("Color Histogram")
    plt.xlabel("Color intensity")
    plt.ylabel("Frequency")
    plt.plot(np.arange(bins), hist)
    plt.show()


def visualize_color_moments(mean, std_dev, skewness):
    labels = ["Mean", "Standard Deviation", "Skewness"]
    values = [mean, std_dev, skewness]

    # Plot each color moment
    plt.figure(figsize=(10, 5))
    for i, (label, value) in enumerate(zip(labels, values)):
        plt.subplot(1, 3, i + 1)
        plt.title(label)
        plt.bar(range(3), value)  # Assuming 3 channels: R, G, B
        plt.xticks(range(3), ["R", "G", "B"])
        plt.tight_layout()
    plt.show()