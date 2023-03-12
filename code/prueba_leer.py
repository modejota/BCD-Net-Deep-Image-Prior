import os
import numpy as np
import time

def scan_files(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                files_list.append(os.path.join(root, file))
    return files_list

def load_images(files_list):
    img_list = []
    for file in files_list:
        img = np.load(file)
        img_list.append(img)
    return img_list

def scan_and_load(path):

    time_list = []
    for _ in range(5):
        start = time.time()
        files_list = scan_files(path)
        end = time.time()
        scan_files_time = end-start
        time_list.append(scan_files_time)
    print("Scan files time: {}".format(np.mean(time_list)))
    print("Found {} files".format(len(files_list)))

    #n_samples = 10000
    #files_list = files_list[:n_samples]
    time_list = []
    for _ in range(5):
        start = time.time()
        img_list = load_images(files_list)
        end = time.time()
        load_images_time = end-start
        time_list.append(load_images_time)
    print("Load images time: {}".format(np.mean(time_list)))
    print("Total time: {}".format(np.mean(time_list)+scan_files_time))    

print("Loading from /data/")
path = "/data/RSNA/test"
scan_and_load(path)

print("Loading from /home/")
path = "/home/fran/prueba_rsna/"
scan_and_load(path)