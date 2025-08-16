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

print('Kinetix camera modes...')
for key in cam.port_speed_gain_table.keys():
    print("{}:{}".format(key, cam.port_speed_gain_table[key]['port_value']))
mode = input('Select camera mode:')
mode = int(mode) # string -> integer
if mode == 0:
    cam_mode = 'Sensitivity'
elif mode == 1:
    cam_mode = 'Speed'
elif mode == 2:
    cam_mode = 'Dynamic_Range'
elif mode == 3:
    cam_mode = 'Sub_Electron'
else:
    print('Camera mode invalid...')

## set camera to appropriate mode
cam.readout_port = int(mode)
cam.speed_table_index = 0
cam.gain = 1

## acquire master bias frame with N short exposures
exp_time = input('Select exposure time [ms]:')
exp_time = int(exp_time) # string -> integer
N = input('Number of dark frames to acquire for averaging:')
N = int(N) # string -> integer

# acquire test frame to define 3D array dimensions
test_frame = cam.get_frame(exp_time=exp_time)
frames = np.empty((N, test_frame.shape[0], test_frame.shape[1]),
                 dtype=test_frame.dtype)

# start the acquisition
cam.start_live(exp_time=exp_time, buffer_frame_count=32, stream_to_disk_path=None)

print('Acquiring frames...')
for n in range(N):
    frame = cam.poll_frame()[0]['pixel_data']
    frames[n] = frame
print('...finished frame collection.')

# average over the acquired frames
master_bias = np.mean(frames, axis=0).astype(np.float32)

# flag any nans if present
if np.isnan(master_bias).any() == True:
    print('nans')

# save a fits and numpy array of the master calibration frame
fitsname = 'calibration_images\master_dark_%s_%sms.fits' % (cam_mode, str(exp_time))
arrname = 'calibration_images\master_dark_%s_%sms.npy' % (cam_mode, str(exp_time))
print('Saving:', fitsname)
save_numpy_as_fits(master_bias, fitsname)
print('Saving:', arrname)
np.save(arrname, master_bias)
