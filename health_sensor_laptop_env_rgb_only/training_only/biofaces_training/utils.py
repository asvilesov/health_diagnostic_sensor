import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

def display_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()
    print("Image format: ", image.format, "\nImage mode: ", image.mode, "\nImage Size: ", image.size)

def display_image_masked(image_path, mask_path):
    image = Image.open(image_path)
    # image = np.array(image)
    mask = Image.open(mask_path)
    
    mask = np.asarray(mask) > 128
    
    image = image.resize((mask.shape[0], mask.shape[1]))
    image = np.array(image)

    print(mask.shape)
    masked_image = np.multiply(image, mask)
    plt.imshow(masked_image)
    plt.show()
    # print("Image format: ", image.format, "\nImage mode: ", image.mode, "\nImage Size: ", image.size)

