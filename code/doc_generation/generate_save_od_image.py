import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import rgb2od, askforimageviaGUI

image_path = askforimageviaGUI()
if image_path is not None:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.array(img)
    img_od = rgb2od(img_array)

    image_path = image_path.split('.')[0]
    image_path = image_path + '_od.jpg'

    plt.imsave(image_path, img_od)
    print(img_array.shape)