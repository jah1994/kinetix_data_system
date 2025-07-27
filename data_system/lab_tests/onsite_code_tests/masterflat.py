import sys
from pyvcam import pvc
from pyvcam.camera import Camera
from astropy.io import fits
import numpy as np

def save_numpy_as_fits(array, filename):
    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)

#b = np.load("data/master_bias_3ms_speed.npy")
print('Loading master dark frame:', sys.argv[1])
b = np.load(sys.argv[1])

pvc.init_pvcam()                   # Initialize PVCAM
cam = next(Camera.detect_camera()) # Use generator to find first camera.
cam.open()

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

# take test exposures to judge the count level
check = 'n'
while check != 'y':
    t_exp = int(input('Input test exposure time [ms]:'))
    frame = cam.get_frame(exp_time=t_exp)
    frame = frame - b
    print('Median pixel value [ADU]:', np.median(frame))
    save_numpy_as_fits(frame, 'test.fits')
    check = input('Happy? [y/n]') # update check


N = input('How many frames to acquire with exposure time %d [ms]' %  t_exp)
N = int(N)

print('Acquiring frames...')
frames = []
cam.start_live(exp_time=t_exp, buffer_frame_count=32, stream_to_disk_path=None)
for n in range(N):
    frame = cam.poll_frame()[0]['pixel_data']
    frame = frame - b # subtract master bias
    print('Frame %d / %d, median pixel value [ADU]: %.3f' % (n+1, N, np.median(frame)))
    frame = frame / np.median(frame) # normalise
    frames.append(frame)
print('...finished frame collection.')

# median average over the frames
frames = np.array(frames)
master_flat = np.median(frames, axis=0).astype(np.float32)

# flag any nans if present
if np.isnan(master_flat).any() == True:
    print('nans')

# save a fits and numpy array of the master calibration frame
fitsname = 'calibration_images\master_flat_%s.fits' % cam_mode
arrname = 'calibration_images\master_flat_%s.npy' % cam_mode
print('Saving:', fitsname)
save_numpy_as_fits(master_flat, fitsname)
print('Saving:', arrname)
np.save(arrname, master_flat)
