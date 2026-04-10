import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("sample.jpg")

# Check if image is loaded
if image is None:
    print("Error: Image not found!")
else:
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 100, 200)

    # Display all images
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Blurred Image")
    plt.imshow(blur, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()