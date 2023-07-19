import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import rgb2od, askforimageviaGUI

image_path = askforimageviaGUI()
if image_path is not None:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to a numpy ndarray
    img_array = np.array(img)
    img_od = rgb2od(img_array)

    # Generate new path to save the image
    image_path = image_path.split('.')[0]
    image_path = image_path + '_od.jpg'

    # Save img_od using matplotlib
    plt.imsave(image_path, img_od)
    
    # Print the shape of the ndarray
    print(img_array.shape)