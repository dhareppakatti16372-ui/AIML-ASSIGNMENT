import cv2
import numpy as np

# Load image (provide your image path)
image = cv2.imread("sample.jpg")

# Check if image loaded
if image is None:
    print("Error: Image not found!")
else:
    # Print shape
    print("Image Shape:", image.shape)
    
    # Height, Width, Channels
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    
    # Print pixel values at a specific point
    x, y = 0, 0
    print(f"\nPixel value at ({x},{y}):", image[x, y])
    
    # Show first 5x5 pixel values
    print("\nTop-left 5x5 pixel values:\n", image[0:5, 0:5])
    
    # Split channels
    blue, green, red = cv2.split(image)
    
    print("\nBlue channel sample:\n", blue[0:5, 0:5])
    print("\nGreen channel sample:\n", green[0:5, 0:5])
    print("\nRed channel sample:\n", red[0:5, 0:5])