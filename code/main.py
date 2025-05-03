import os
import sys
sys.path.append('python')
from OCT_Dewarp_BL import OCT_Dewarp_BL

if __name__ == "__main__":
    original_images_path = "../images/original"

    # List all files in the original images directory
    original_images = []
    for filename in os.listdir(original_images_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            original_images.append(os.path.join(original_images_path, filename))

    # Print the list of original images
    print("Original images:")
    for image in original_images:
        print(image) 

    # Dewarp each image
    for image in original_images:
        _ = OCT_Dewarp_BL(image, debug=False)
        print(f"Dewarped image {image} saved in ../images/python_dewarped")
