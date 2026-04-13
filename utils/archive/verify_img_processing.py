# this script was written to verify if image_processing.py works as intended
# this script shows the original image and it's representation in frequency domain

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_magnitude_spectrum(image_path):
    """
    converts an image to grayscale and returns its frequency magnitude spectrum
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # compute 2d fourier transform and shift zero to centre
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    
    # calculate magnitude specturum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum

def verify_outputs(original_path, fpr_path, comp_path, output_filename="verification_plot.png"):
    """
    saves the frequency domains of the three images side-by-side
    """
    spec_orig = get_magnitude_spectrum(original_path)
    spec_fpr = get_magnitude_spectrum(fpr_path)
    spec_comp = get_magnitude_spectrum(comp_path)

    if any(s is None for s in [spec_orig, spec_fpr, spec_comp]):
        print("Error: One or more images could not be loaded. check file paths.")
        return

    plt.figure(figsize=(18, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(spec_orig, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(spec_fpr, cmap="gray")
    plt.title("Fingerprint Removed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(spec_comp, cmap="gray")
    plt.title("Compressed")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_filename)
    print(f"Success: plot saved as '{output_filename}'")

if __name__ == "__main__":
    archive_dir = os.path.join("utils", "archive")
    original = os.path.join(archive_dir, "test_input.png")
    fingerprint_removed = os.path.join(archive_dir, "test_fingerprint_removed.png")
    compressed = os.path.join(archive_dir, "test_compressed.jpg")
    output_plot = os.path.join(archive_dir, "verification_plot.png")
    
    if os.path.exists(original) and os.path.exists(fingerprint_removed):
        print(f"Images found in '{archive_dir}', generating frequency plot...")
        verify_outputs(original, fingerprint_removed, compressed, output_filename=output_plot)
    else:
        print(f"Error: Could not find '{original}' or '{fingerprint_removed}'.")