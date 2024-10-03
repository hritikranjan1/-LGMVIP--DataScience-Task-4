import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_to_sketch(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_img = 255 - gray_img

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(inverted_gray_img, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred_img = 255 - blurred_img

    # Create the pencil sketch image by mixing the grayscale image with the inverted blurred image
    sketch_img = cv2.divide(gray_img, inverted_blurred_img, scale=256)

    return sketch_img

def display_images(original, sketch):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Pencil Sketch")
    plt.imshow(sketch, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'  # Change this to your image path
    original_image = cv2.imread(image_path)
    sketch_image = image_to_sketch(image_path)

    display_images(original_image, sketch_image)
