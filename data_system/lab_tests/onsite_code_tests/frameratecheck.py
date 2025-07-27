from pyvcam import pvc
from pyvcam.camera import Camera
from astropy.io import fits
import numpy as np
import time

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

## check frame acquisition speed in this mode
N = 1000

times = np.zeros(N)
texps = np.zeros(N) # housekeeping... useful diagnostic
seqs = np.zeros(N) # housekeeping... sequence number

exp_time = input('Select exposure time [ms]:')
exp_time = int(exp_time)
cam.start_live(exp_time=exp_time, buffer_frame_count=1, stream_to_disk_path=None)

# start frame rate check
t0 = time.perf_counter()
for n in range(N):

    # oldest frame in buffer popped from the queue
    #data = cam.poll_frame()[0]['pixel_data'] # TODO time information???
    frame = cam.poll_frame(copyData=False)

    # store time exposure acquired
    times[n] = time.perf_counter() - t0

    # exptime (housekeeping)
    texps[n] = frame[1]

    # sequence number (housekeeping)
    seqs[n] = frame[2]

t_total = time.perf_counter() - t0

# return camera to normal state
cam.finish()
print('Finished live acquistion in %.3f seconds.' % t_total)
print('Frame-rate (Hz):', N / t_total)
print('Average time available to process a frame (ms):', 1000 * t_total / N)
