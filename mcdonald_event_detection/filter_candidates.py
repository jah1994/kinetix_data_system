# imports
from tqdm import tqdm # progressbar
import os
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# sigma threshold
sigma_thresh = 6

# find SNR peak and image files
path = "D:\\McDonaldObs_Feb_2024\\Analysis_PCA\\Analysis_20th_Feb_v2\\"
file_paths, img_paths = [], []
for RUN in range(0, 100):
    for SCENE in range(0, 100):
        file_path = os.path.join(path, 'RUN_' + str(RUN), 'SCENE_' + str(SCENE), 'analysis', 'results', 'res_local_peaks.npy')
        if os.path.exists(file_path):
            img_path = os.path.join(path, 'RUN_' + str(RUN), 'SCENE_' + str(SCENE), 'analysis', 'closeups')
            file_paths.append(file_path)
            img_paths.append(img_path)

print('File paths:', file_paths)


# res_local_peaks.append([i, snr_maxs, loc_check, t_ids])
# identify isolated peaks in each file
for file_path, images in zip(file_paths, img_paths):

    print(file_path, images)

    # load the result file
    res = np.load(file_path, allow_pickle=True)

    # all locations
    peak_locs = []
    source_ids = []
    peak_snrs = []
    for i in range(len(res)):
        source, peaks, locations, templates = res[i]
        for loc, snr in zip(locations, peaks):
            peak_locs.append(loc)
            source_ids.append(source)
            peak_snrs.append(snr)


    # find isolated peaks
    window = 500 # 1875
    peak_locs = np.array(peak_locs)
    for i,(peak, source, snr) in enumerate(zip(peak_locs, source_ids, peak_snrs)):

        # filter by snr
        if snr < sigma_thresh:
            continue

        # remove the particular peak
        peak_locs_ = np.delete(peak_locs, i)
        diff = np.abs(peak - peak_locs_)

        # is the peak sufficiently isolated?
        if np.all(diff > window) == True:
            #print(source, peak)
            #print('Isolated peak!')

            # Load the .png image file
            #subdir = file.split('/')[-1].split('_peaks')[0].split('results\\')[-1]
            img_path = os.path.join(images, str(source) + '_' + str(peak) + '.png')
            try:
                img = mpimg.imread(img_path)

                def on_key(event):

                    if event.key == 't':
                        # Save the current figure
                        file_name = path.split('\\')[-2] + '_' + img_path.split('\\')[4] + '_' + img_path.split('\\')[5] + '_' +  img_path.split('\\')[-1]
                        plt.savefig('D:/test_images/' + file_name)
                        print("Saving file:", file_name)
                        plt.close()
                    elif event.key == 'q':
                        # Quit the application
                        plt.close()
                    else:
                        # Ignore other keys
                        pass

                # Plot the image
                plt.figure(figsize=(20,20))
                plt.imshow(img)
                plt.axis('off')  # Optional: Turn off the axes
                plt.title(file_path)
                plt.show(block=False)

                # Attach the on_key function to handle keyboard events
                fig = plt.gcf()
                fig.canvas.mpl_connect('key_press_event', on_key)

                plt.show()


            except Exception as e:
                print(e)
