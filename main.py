import cv2
import numpy as np
from scipy.ndimage import convolve

# Load the image from file
image_path = 'window.png'

# Get file name without .png
image_name = image_path.split('.')[0]

image = cv2.imread(image_path)

# Convert the image to floating point representation for processing
image = image.astype(np.float32) / 255

# Define the parameter for the fuzzy model (using an example value, needs to be adjusted based on noise variance)
a1 = 15  # This is an example parameter that would need to be tuned for the actual noise level

# Define the fuzzy membership function for "small" distances
def mu_sm(d, a1):
    if d >= 4 * a1:
        return 0
    elif d <= a1:
        return 1
    else:
        return 1 - 3 * (d - a1) / (4 * a1 - d)

# Define a function to perform fuzzy model-based vector smoothing
def fuzzy_vector_smoothing(image, a1):
    # Get the dimensions of the image
    rows, cols, channels = image.shape

    # The new image after smoothing
    new_image = np.zeros_like(image)

    # Define the window size for the local neighborhood
    window_size = 3  # Using a 3x3 window

    # Pad the image to handle the borders
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Process each pixel in the image
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Initialize the sum and weight for each channel
            sum_weights = np.zeros(channels)
            sum_pixels = np.zeros(channels)

            # Go through the local neighborhood
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Get the current neighbor pixel
                    neighbor = padded_image[i + k, j + l]

                    # Calculate the Euclidean distance between the current pixel and the neighbor
                    d = np.linalg.norm(image[i - 1, j - 1] - neighbor)

                    # Get the weight from the fuzzy membership function
                    weight = mu_sm(d, a1)

                    # Accumulate the weighted sum
                    sum_weights += weight
                    sum_pixels += weight * neighbor

            # Calculate the new pixel value as the weighted average
            new_pixel = sum_pixels / sum_weights
            new_image[i - 1, j - 1] = new_pixel

    return new_image

# Apply the fuzzy vector smoothing to the image
smoothed_image = fuzzy_vector_smoothing(image, a1)

# Convert the processed image back to uint8
smoothed_image = (smoothed_image * 255).astype(np.uint8)

name = f'smoothed_{image_name}_a1={a1}.png'

# Save the smoothed image
cv2.imwrite(name, smoothed_image)

# Function to apply Fuzzy Model-Based Sharpening
def fuzzy_model_based_sharpening(luminance, a2, a3, lambd):
    # Placeholder for the sharpening process - the actual implementation would go here.
    # Convert luminance channel to float for processing
    lum_float = luminance.astype(np.float32)
    
    # Placeholder array for the sharpened luminance channel
    sharpened_lum = np.zeros_like(lum_float)

    # Define the window size for local sharpening (3x3)
    window_size = 3

    # Padding the luminance channel to handle borders
    padded_lum = np.pad(lum_float, ((1, 1), (1, 1)), 'reflect')

    # Compute the high-pass filter output for each pixel
    for i in range(1, padded_lum.shape[0] - 1):
        for j in range(1, padded_lum.shape[1] - 1):
            # Extract the local 3x3 block
            local_block = padded_lum[i-1:i+2, j-1:j+2].reshape(window_size**2, 1)
            central_pixel_lum = padded_lum[i, j]

            # Calculate the luminance differences from the central pixel
            lum_differences = local_block - central_pixel_lum

            # Apply the fuzzy membership function to the differences
            mu_LA = np.piecewise(
                lum_differences,
                [np.abs(lum_differences) <= a2,
                 (np.abs(lum_differences) > a2) & (np.abs(lum_differences) <= a3),
                 (np.abs(lum_differences) > a3) & (np.abs(lum_differences) <= 2*a3 - a2)],
                [0, lambda x: 1 - (np.abs(x) - a2) / (2*(a3 - a2)), lambda x: (4*a3 - np.abs(x)) / (3*a2 - 2*a3), 1]
            )

            # Calculate the high-pass filter output
            high_pass_output = np.sum(mu_LA * lum_differences) / 8
            
            # Combine the original luminance values with the high-pass filter output
            # constrained by the minimum function and scaled by lambd
            sharpened_lum[i-1, j-1] = min(central_pixel_lum, central_pixel_lum + lambd * high_pass_output)

    # Clip values to maintain valid range
    sharpened_lum = np.clip(sharpened_lum, 0, 255)

    return sharpened_lum.astype(np.uint8)

# Load the smoothed image from the previous step
smoothed_image_path = name
smoothed_image = cv2.imread(smoothed_image_path)

# Convert to YCbCr color space
ycbcr_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycbcr_image)

# Define the parameters for the fuzzy model
a2 = 20
a3 = 150
lambd = 2
# Apply the Fuzzy Model-Based Sharpening
sharpened_y = fuzzy_model_based_sharpening(y, a2, a3, lambd)

# Merge the sharpened Y channel back with the original Cr and Cb channels
sharpened_ycbcr = cv2.merge([sharpened_y, cr, cb])

# Convert back to BGR color space
sharpened_image = cv2.cvtColor(sharpened_ycbcr, cv2.COLOR_YCrCb2BGR)

# Save and show the sharpened image path
sharpened_image_path = f'sharpened_{image_name}_{a1}_{a2}_{a3}_{lambd}.png'
cv2.imwrite(sharpened_image_path, sharpened_image)

