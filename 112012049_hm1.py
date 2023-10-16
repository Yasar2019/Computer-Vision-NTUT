from PIL import Image
import numpy as np
import cv2

def convert_to_grayscale(img_array):

    # Check if the image has an alpha channel (transparency)
    has_alpha = img_array.shape[2] == 4

    # Convert to grayscale using the averaging method
    gray_values = img_array[:, :, :3].mean(axis=2)
    gray_image_array = np.repeat(gray_values[:, :, np.newaxis], 3, axis=2)

    # If the image has an alpha channel, append it back to the grayscale image
    if has_alpha:
        gray_image_array = np.concatenate(
            [gray_image_array, img_array[:, :, 3:]], axis=2)

    # Convert back to an image object
    gray_image = Image.fromarray(np.uint8(gray_image_array))
    gray_image.save("results/taipei101_Q1.png")

    # Returning the grayscale values as a 2D array
    return gray_image_array[:, :, 0]



def convolution_operation(image_array, kernel):
    """
    Perform convolution operation on the input image with a specified kernel.
    """
    i_height, i_width = image_array.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2
    padded_image = np.pad(
        image_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image_array)
    for i in range(i_height):
        for j in range(i_width):
            region = padded_image[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(region * kernel)
    return output


def apply_edge_detection_convolution(image_array):
    """
    Apply edge detection convolution using the Sobel operator.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = convolution_operation(image_array, sobel_x)
    gy = convolution_operation(image_array, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    edge_image_array = (magnitude / magnitude.max() * 255).astype(np.uint8)
    edge_image = Image.fromarray(edge_image_array)
    edge_image.save("results/taipei101_Q2.png")
    return edge_image_array




def max_pooling(image_array, kernel_size=2, stride=2):
    """
    Apply max pooling on the input image.
    """
    i_height, i_width = image_array.shape
    o_height = (i_height - kernel_size) // stride + 1
    o_width = (i_width - kernel_size) // stride + 1
    output = np.zeros((o_height, o_width))
    for i in range(0, o_height):
        for j in range(0, o_width):
            region = image_array[i*stride:i*stride +
                                 kernel_size, j*stride:j*stride+kernel_size]
            output[i, j] = np.max(region)
    pooled_image = Image.fromarray(
        (output / output.max() * 255).astype(np.uint8))
    pooled_image.save("results/taipei101_Q3.png")
    return output


def binarization(image_array, threshold=128):
    """
    Binarize the input image based on a specified threshold.
    """
    binarized_array = np.where(image_array >= threshold, 255, 0)
    binarized_image = Image.fromarray(binarized_array.astype(np.uint8))
    binarized_image.save("results/taipei101_Q4.png")
    return binarized_array


def otsu_threshold(image_array):
    """
    Calculate the optimal threshold value using Otsu's method.
    """
    # Compute the histogram of the image
    hist, bin_edges = np.histogram(image_array, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute the cumulative sums (normalized)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Compute between-class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Find the threshold that maximizes the between-class variance
    idx = np.argmax(variance)
    optimal_threshold = bin_centers[:-1][idx]

    return optimal_threshold


def apply_operations(image_path):
    """Processes the image through the entire pipeline."""
    # Load the image
    img = cv2.imread(image_path)
    img_array = np.array(img)

    gray_array = convert_to_grayscale(img_array)
    optimal_thresh = otsu_threshold(gray_array)
    edge_array = apply_edge_detection_convolution(gray_array)
    pooled_array = max_pooling(edge_array)
    binarized_array = binarization(pooled_array, threshold=optimal_thresh)

    return Image.fromarray(binarized_array.astype(np.uint8))


# Use the function to process the image and display the result
result_image = apply_operations("test_img/taipei101.png")
result_image.show()
