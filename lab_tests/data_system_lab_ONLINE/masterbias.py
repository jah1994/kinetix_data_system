from pyvcam import pvc
from pyvcam.camera import Camera
from astropy.io import fits
import numpy as np
import time

def save_numpy_as_fits(array, filename):
    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)


pvc.init_pvcam()                   # Initialize PVCAM
cam = next(Camera.detect_camera()) # Use generator to find first camera.
cam.open()

for key in cam.port_speed_gain_table.keys():
    print("{}:{}".format(key, cam.port_speed_gain_table[key]['port_value']))


mode = input('Select camera mode:')

## set camera to appropriate mode
cam.readout_port = int(mode)
cam.speed_table_index = 0
cam.gain = 1

## acquire master bias frame with 1000 short exposures
exp_time = 3
test_frame = cam.get_frame(exp_time=exp_time)

N = 5000
n = 0

frames = np.empty((N, test_frame.shape[0], test_frame.shape[1]),
                 dtype=test_frame.dtype)

#exp_time = 2 # 2ms i.e. 500 Hz
cam.start_live(exp_time=exp_time, buffer_frame_count=32, stream_to_disk_path=None)

print('Acquiring frames...')
while n < N:

    frame = cam.poll_frame()[0]['pixel_data']
    frames[n] = frame

    n += 1

print('...finished frame collection.')

master_bias = np.mean(frames, axis=0).astype(np.float32)

if np.isnan(master_bias).any() == True:
    print('nans')

fitsname = 'master_bias_3ms_speed_take2.fits'
arrname = 'master_bias_3ms_speed_take2.npy'
save_numpy_as_fits(master_bias, fitsname)
np.save(arrname, master_bias)
