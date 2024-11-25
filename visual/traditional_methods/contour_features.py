import cv2
import numpy as np
from mahotas.zernike import zernike_moments
import matplotlib.pyplot as plt

def compute_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)

def compute_fourier_descriptors(contour):
    contour = contour.squeeze()
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    descriptors = np.fft.fft(complex_contour)
    return descriptors


def compute_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def compute_zernike_moments(image, radius=21):
    return zernike_moments(image, radius)


def display_contour(image, contour):
    output_image = image.copy()
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
    plt.imshow(output_image, cmap="gray")
    plt.axis("off")
    plt.show()


def display_fourier_descriptors(descriptors):
    plt.plot(descriptors.real, descriptors.imag, "o-")
    plt.title("Fourier Descriptors")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.axis("equal")
    plt.show()


def display_hu_moments(hu_moments):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(hu_moments)), hu_moments)
    plt.title("Hu Moments")
    plt.xlabel("Moment")
    plt.ylabel("Value")
    plt.show()


def display_zernike_moments(zernike_moments):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(zernike_moments)), np.abs(zernike_moments))
    plt.title("Zernike Moments")
    plt.xlabel("Moment")
    plt.ylabel("Value")
    plt.show()