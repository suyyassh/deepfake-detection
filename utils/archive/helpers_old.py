# this script has the same functions as preprocessing/helpers.py
# it was written first and was used to check if the functions work as intended by generating and saving outputs (see the quick test at the bottom)

import cv2
import numpy as np
import os

def apply_compression(input_path, output_path, quality=12, scale_factor=0.5):
    """
    applies compression to images
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read the image at '{input_path}")
        return False
    
    height, width = img.shape[:2]

    # downscaling the image - immediate loss of pixels
    small_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_AREA)

    # upscaling the image
    blurred_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(output_path, blurred_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return True

def remove_noise(input_path, output_path, cutoff_freq=40):
    """
    a low pass filter that removes noise
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read the image at '{input_path}'")
        return False
    
    # split R, G and B channels for processing
    channels = cv2.split(img)
    processed_channels = []

    for ch in channels:

        # transform to frequency domain and shift low frquencies to the centre
        f_transform = np.fft.fft2(ch)
        f_shift = np.fft.fftshift(f_transform) ###########

        # finding the coordinates of the centre
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2

        # create a blank grid and find the distance of each pixel from the centre
        Y, X = np.ogrid[:rows, :cols]
        dist_squared = (X - ccol)**2 + (Y - crow)**2

        # applying the gaussian formula
        mask = np.exp(-dist_squared / (2 * (cutoff_freq**2)))

        # applying a smooth mask to the frequency data
        f_shift_filtered = f_shift * mask

        # inverse transform back to spatial domain
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)

        # extract the real part and clip to valid pixel range
        img_back_clean = np.clip(img_back.real, 0, 255).astype(np.uint8)
        processed_channels.append(img_back_clean)

    # merge channels and save
    final_img = cv2.merge(processed_channels)
    cv2.imwrite(output_path, final_img)
    return True

# quick test
# to test - place an image named 'test_input.png' in utils/arhive
if __name__ == "__main__":
    archive_dir = os.path.join("utils", "archive")
    img = os.path.join(archive_dir, "test_input.png")
    comp_img = os.path.join(archive_dir, "test_compressed.jpg")
    fpr_img = os.path.join(archive_dir, "test_fingerprint_removed.png")
    
    if os.path.exists(img):
        print("Compressing image...")
        apply_compression(img, comp_img)
        print("Removing fingerprints...")
        remove_noise(img, fpr_img, cutoff_freq=40)
        print("Success: check utils/archive for results")
    else:
        print(f"Warning: to test, place an image named 'test_input.png' into utils/archive")