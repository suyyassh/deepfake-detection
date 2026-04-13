# this script provides the helper functions to compress an image and remove noise

import cv2
import numpy as np

def apply_compression(img, quality=12, scale_factor=0.5):
    """
    applies compression to images
    """
    if img is None:
        return None

    height, width = img.shape[:2]

    # downscaling the image - immediate loss of pixels
    small_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_AREA)

    # upscaling the image
    blurred_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # encode the image and return
    _, encimg = cv2.imencode('.jpg', blurred_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encimg, 1)

def remove_noise(img, cutoff_freq=40):
    """
    a low pass filter that removes noise
    """
    if img is None:
        return None
    
    # split R, G and B channels for processing
    channels = cv2.split(img)
    processed_channels = []

    for ch in channels:

        # transform to frequency domain and shift low frquencies to the centre
        f_transform = np.fft.fft2(ch)
        f_shift = np.fft.fftshift(f_transform)

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

    # merge channels and return
    return cv2.merge(processed_channels)