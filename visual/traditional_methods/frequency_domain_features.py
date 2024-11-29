import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt


def compute_dft(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    )
    return magnitude_spectrum

def visualize_dft(magnitude_spectrum):
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum")
    plt.colorbar()
    plt.show()

def compute_dwt(image):
    coeffs = pywt.dwt2(image, "haar")
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def visualize_dwt(cA, cH, cV, cD):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(cA, cmap="gray")
    axes[0, 0].set_title("Approximation Coefficients (cA)")

    axes[0, 1].imshow(cH, cmap="gray")
    axes[0, 1].set_title("Horizontal Detail Coefficients (cH)")

    axes[1, 0].imshow(cV, cmap="gray")
    axes[1, 0].set_title("Vertical Detail Coefficients (cV)")

    axes[1, 1].imshow(cD, cmap="gray")
    axes[1, 1].set_title("Diagonal Detail Coefficients (cD)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def apply_gabor_filter(image, frequency=0.6):
    from skimage.filters import gabor

    real, imag = gabor(image, frequency=frequency)
    return real, imag

def visualize_gabor(real, imag):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(real, cmap="gray")
    plt.title("Real Part of Gabor Filter")

    plt.subplot(1, 2, 2)
    plt.imshow(imag, cmap="gray")
    plt.title("Imaginary Part of Gabor Filter")

    plt.tight_layout()
    plt.show()
