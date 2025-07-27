from pyvcam import pvc
from pyvcam.camera import Camera
from astropy.io import fits
import numpy as np

def save_numpy_as_fits(array, filename):
    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)

b = np.load("data/master_bias_3ms_speed.npy")

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

frames = []

print('Acquiring frames...')
cam.start_live(exp_time=t_exp, buffer_frame_count=32, stream_to_disk_path=None)
n = 0
while n < N:

    frame = cam.poll_frame()[0]['pixel_data']
    frame = frame - b # subtract master bias
    print('Frame %d / %d, median pixel value [ADU]: %.3f' % (n+1, N, np.median(frame)))
    frame = frame / np.median(frame) # normalise
    frames.append(frame)
    n += 1

print('...finished frame collection.')

frames = np.array(frames)
master_flat = np.median(frames, axis=0).astype(np.float32)

if np.isnan(master_flat).any() == True:
    print('nans')

fitsname = 'master_flat_speed_improved.fits'
arrname = 'master_flat_test_speed_improved.npy'
save_numpy_as_fits(master_flat, fitsname)
np.save(arrname, master_flat)
