################# imports ########################
import config # software configuration file
import os
import sys
import logging

# usage example in terminal
#python run.py 0

# scene id
SCENE = int(sys.argv[1]) # no data value of rasters


try:
    from pyvcam import pvc
    from pyvcam.camera import Camera
except ModuleNotFoundError:
    print('Unable to locate PyVCAM module... camera functionality not available.')


# custom image processing functions
import imageproc

# standard libraries
import numpy as np
import time
from astropy.stats import mad_std
from astropy.io import fits
from photutils.detection import find_peaks
from skimage.io import MultiImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
##########################################################


# suppress numba warning re reflected lists
if config.suppress_numba_warning is True:
    from numba.core.errors import NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# useful function to save arrays to fits file format
def save_numpy_as_fits(numpy_array, filename):
    hdu = fits.PrimaryHDU(numpy_array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)



# dubugging and housekeeping info written to housekeeping.log
out_path = config.out_path + '_scene_' + str(SCENE)

# if out_path doesn't exist, make it
if os.path.exists(out_path) == False:
    print('Making result directory:', out_path)
    os.mkdir(out_path)
else:
    print('Result directory already exists:', out_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[
        logging.FileHandler(os.path.join(out_path, "housekeeping.log")),
        logging.StreamHandler()
    ])


### Online vs offline mode ####
## For testing purposes, it is useful to run the software offline on pre-acquired data
## ONLINE = True : Live mode acquistion with Kinetix
## ONLINE = False : (Testing onlyl) Run offline on pre-acquired imaging data
ONLINE = config.online


## grab calibration frames - N.B. The calibration frames for the Kinetix need to be
## acquired using the same mode of operation e.g. Dynamic, Speed etc.
logging.info('Loading calibration frames...')
flat = np.load(config.flat)
dark = np.load(config.dark)
logging.info('Found flat and dark frames.')


if ONLINE is True:

    ## Initialise and open the camera
    logging.info('Initialising the camera...')
    pvc.init_pvcam()                   # Initialize PVCAM
    cam = next(Camera.detect_camera()) # Use generator to find first camera.
    cam.open()                         # Open the camera.
    logging.info('Camera open!')

    ## set camera mode
    if config.sensitivity is True:
        cam.readout_port = 0
        cam.speed_table_index = 0
        cam.gain = 1
        logging.info('Camera in Sensitivity mode.')
    elif config.speed is True:
        cam.readout_port = 1
        cam.speed_table_index = 0
        cam.gain = 1
        logging.info('Camera set to Speed mode.')
    elif config.dynamic is True:
        cam.readout_port = 2
        cam.speed_table_index = 0
        cam.gain = 1
        logging.info('Camera in Dynamic Range mode.')
    else:
        cam.readout_port = 0
        cam.speed_table_index = 0
        cam.gain = 1
        logging.info('No camera mode specified, defaulting to Sensitivity mode...')


    ### Acquire reference frame - return source positions and aperture size
    ref_exp_time = config.ref_exp_time
    logging.info('Acquring reference frame with a %.2f second integration', (ref_exp_time / 1000))
    ref = cam.get_frame(exp_time=ref_exp_time)
    logging.info('Done!')

else:
    logging.info('Running software offline...')
    ref = fits.getdata(os.path.join(config.offline_path, config.offline_ref))
    ref = ref.astype(dark.dtype) # change to numpy dtype
    logging.info('Reference frame loaded.')


logging.info('Flat correcting and dark subtracting reference...')
ref = imageproc.proc(ref, dark, flat)
logging.info('Done!')

# compute estimates of the noise and sky level
sky = np.median(ref)
logging.info('Reference sky level [ADU]: %.3f', sky)
ref -= sky # sky subtract
std = mad_std(ref)
logging.info('Reference MAD [ADU]: %.3f', std)

# save reference for visual inspection
save_numpy_as_fits(ref, os.path.join(out_path, 'ref.fits'))
logging.info('Saved the reference image as a ref.fits file.')


logging.info('Starting source detection routine...')
# run initial peak finding routine on data to detect bright stars
peaks = find_peaks(ref, threshold=10*std, box_size=25, border_width=50)
peaks.sort('peak_value')
peaks.reverse() # sort so that the brightest star is first

# fit 2D Gaussian to the bright stars to estimate stamp radiu in x and y
logging.info('Estimating PSF model and stamp radii')
r = config.r0 # initial guess for stamp radius
rx, ry, sigma_x, sigma_y, psf_model = imageproc.update_r(ref, peaks, r=r, nsigma=config.nsigma, lim=config.lim, out_path=out_path)

logging.info('sigma_x=%.3f, sigma_y=%.3f', sigma_x, sigma_y)
logging.info('rx=%d, ry=%d:', rx, ry)

# match-filter reference with the PSF Model and normalise to generate a detection map
rdnoise = config.rdnoise
gain = config.gain
if config.emp_detect is True:
    D_norm = imageproc.make_detection_map_empirical(ref, psf_model, std)
else:
    D_norm = imageproc.make_detection_map(ref, psf_model, rdnoise, gain)
save_numpy_as_fits(D_norm, os.path.join(out_path, 'D_norm.fits')) # save for visual inspection
#D_norm = fits.getdata('D_norm.fits')

## run a peak finding routine on the normalised detection map
positions = find_peaks(D_norm, threshold=config.thresh, box_size=rx, border_width=2*rx)
positions.sort('peak_value')
positions.reverse() # sort so that the highest SNR* star is first (Detection map is normalised by uncertainties)
peaks = positions['peak_value'] # flux peaks
positions = np.vstack((positions['x_peak'], positions['y_peak'])).T # numba doesn't like astropy tables
nsources = len(positions)
logging.info('Detected sources: %d', nsources)

'''
print('TESTING: bypasssing source detection...')
N = 128
positions = np.random.uniform(100, 3100, (N, 2)).astype(int)
nsources = len(positions)
rx, ry = 30, 30
rdnoise = config.rdnoise # e-
gain = config.gain # e-/ADU
'''

# plot Detection map and visually check detections look good
ls = 15 # tick label size
fs = 12 # text fontsize

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=ls)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(D_norm, origin='lower')
cbar = ax.figure.colorbar(im, cax=cax)
cbar.set_label('$D / \sigma_{D}$', fontsize=ls)
cbar.ax.tick_params(labelsize=ls)
#fig.colorbar(im, cax=cax, orientation='vertical')
for p,pos in enumerate(positions):
    ax.text(pos[0], pos[1], str(p), fontsize=fs)
plt.savefig(os.path.join(out_path, 'D_norm.png'), bbox_inches='tight') # save to disk for reference
#plt.show();


### automated background region selection
nboxes = config.nbboxes
logging.info('Number of background region boxes: %d', nboxes)

# KDE of source positions, weighted by brightness, to allow a constrained search in sparsely populated regions
bb_pos, bb_rs = imageproc.background_boxes(positions, peaks, ref, rx, ry, out_path=out_path, N=nboxes)

# plot template and visually check apertures look OK
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=ls)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(ref + sky, origin='lower')
cbar = ax.figure.colorbar(im, cax=cax)
cbar.set_label("Counts [ADU]", fontsize=ls)
cbar.ax.tick_params(labelsize=ls)
#fig.colorbar(im, cax=cax, orientation='vertical')
for p,pos in enumerate(positions):
    ax.add_patch(patches.Rectangle(xy=(pos[0]-rx, pos[1]-ry),
                                   width=2*rx, height=2*ry, fill=False,
                                   label=p))
    ax.text(pos[0], pos[1], str(p), fontsize=fs)

for j, (pos, rs) in enumerate(zip(bb_pos, bb_rs)):

    ax.add_patch(patches.Rectangle(xy=(pos[0] - rs[0], pos[1] - rs[1]),
                                   width=2*rs[0], height=2*rs[1], fill=False, label=j, color='red'))
    ax.text(pos[0], pos[1], str(j), fontsize=fs, c='r')
plt.savefig(os.path.join(out_path, 'ref_annotated.png'), bbox_inches='tight')
#plt.show()

#### Housekeeping data ###
np.save(os.path.join(out_path, 'positions.npy'), positions) # save positions of sources
np.save(os.path.join(out_path, 'bb_pos.npy'), bb_pos) # save backbround box positions...
np.save(os.path.join(out_path, 'bb_rs.npy'), bb_rs) #... and their raddii

# generate calibration frame stamp_size
dark_stamps, flat_stamps = imageproc.calibration_stamps(np.copy(dark), np.copy(flat), positions, rx, ry)
sky_dark_stamps, sky_flat_stamps = imageproc.sky_calibration_stamps(np.copy(dark), np.copy(flat), bb_pos, bb_rs)

############ Live mode acquistion #################
logging.info('JIT compiling functions...')

phot = imageproc.fluxes_stamps(ref, dark_stamps, flat_stamps, positions, nsources, rx, ry)
#phot, phot_var = imageproc.fluxes_and_uncertainties_stamps(ref, dark_stamps, flat_stamps, positions, nsources, rx, ry, rdnoise, gain)
skys = imageproc.sky_stamps(ref, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
logging.info('Done!')
####
pf = config.plot_freq # plot frequency / stamp save frequency

#### Real Time plotting / Saving source stamps
logging.info('Select a source to track:')
s1 = int(input('Source 1:'))
logging.info('Tracking source %d' % s1)

if config.real_time_plot is True:

    fig = plt.figure(figsize=(2, 2))

    # power-law rescaling of stamp data
    pow = config.pow

    ax1 = fig.add_subplot(1, 1, 1) # image

    RX, RY = np.linspace(0, 2 * rx, num= 2 * rx), np.linspace(0, 2* ry, num= 2 * ry)
    X, Y = np.meshgrid(RX, RY)

    minview = int(np.median(dark))
    c = 100
    img1 = ax1.imshow(X, cmap="Greys", vmin=minview, vmax=minview+c, origin='lower')
    ax1.set_title('Source: %d' % s1)

    fig.canvas.draw()

    # cache the background (blitting)
    ax1background = fig.canvas.copy_from_bbox(ax1.bbox)

    plt.show(block=False)


if ONLINE is True:
    ## setup a live mode acquistion - frames stored in buffer
    logging.info('Starting live mode acquistion...')
    exp_time = config.exp_time # ms
    logging.info('Exposure time [ms]: %f', exp_time)
    cam.start_live(exp_time=exp_time, buffer_frame_count=config.buffer_count, stream_to_disk_path=None)
    logging.info('Camera now collecting data...')

else:
    ## offline mode - run on pre-acquired imaging data
    path = config.offline_path
    files = [f for d, s, f in os.walk(path)][0]

    # order data files
    ordered_files = []
    scale = np.arange(0, 1000).astype(str)
    for s in scale:
        for f in files:
            if 'ss_stack' in f and '_' + s + '.tiff' in f:
                ordered_files.append(f)

t0 = time.perf_counter()

# number of data batches: total number of images processed = batches * N
batches = config.batches

for batch in range(batches):

    if ONLINE is False:
        file_name = os.path.join(path, ordered_files[batch])
        print('Loading data from %s' % file_name)
        img_coll = np.array(MultiImage(file_name))[0]
        print(img_coll.shape, img_coll.dtype)
        print('Done!')

    t0_batch = time.perf_counter()

    N = config.N # total number of exposures to acquire per batch

    # generate array to store results
    photometry = np.zeros((N, len(positions)))
    #photometry_var = np.zeros((N, len(positions)))
    times = np.zeros(N)
    texps = np.zeros(N) # housekeeping... useful diagnostic
    seqs = np.zeros(N) # housekeeping... sequence number
    sky_lvls = np.zeros((N, len(bb_pos)))
    processing_times = np.zeros(N)

    n = 0 # counter for number of acquired frames

    ## initialise arrays to hold stamps for real-time plotting
    stamp1s = np.zeros((pf, 2 * ry, 2 * rx))
    stamp_count = 0

    ## initialise array for autoguiding stamp
    autoguide_stamps = np.zeros((pf, config.ag_stamp_size, config.ag_stamp_size))
    ag_r = int(config.ag_stamp_size / 2)

    while n < N:

        try:

            if ONLINE is True:
                # oldest frame in buffer popped from the queue
                frame = cam.poll_frame(copyData=False)

                # store time exposure acquired
                times[n] = time.perf_counter() - t0

                # exptime (housekeeping)
                texps[n] = frame[1]

                # sequence number (housekeeping)
                seqs[n] = frame[2]

                # pull the data
                data = frame[0]['pixel_data']

            else:
                # locate appropriate file - batch equivalent to imaging stack in this instance
                #file_name = os.path.join(path, ordered_files[batch])
                #data = MultiImage(file_name)[n] # load the nth image in the stack/batch
                data = img_coll[n]

            ### The workhorse ####
            tproc = time.perf_counter()

            # aperture photometry
            phot = imageproc.fluxes_stamps(data, dark_stamps, flat_stamps, positions, nsources, rx, ry)
            #phot, phot_var = imageproc.fluxes_and_uncertainties_stamps(data, dark_stamps, flat_stamps, positions, nsources, rx, ry, rdnoise, gain)
            photometry[n] = phot

            # sky regions
            skys = imageproc.sky_stamps(data, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
            sky_lvls[n] = skys

            proc_time = 1000 * (time.perf_counter() - tproc)
            processing_times[n] = proc_time

            #print('Proc time [ms]:', proc_time)

            ## cache source stamps
            stamp1s[stamp_count] = data[positions[s1][1] - ry : positions[s1][1] + ry, positions[s1][0] - rx : positions[s1][0] + rx]

            ## cache autoguide stampn
            autoguide_stamps[stamp_count] = data[positions[s1][1] - ag_r : positions[s1][1] + ag_r, positions[s1][0] - ag_r : positions[s1][0] + ag_r]

            stamp_count += 1

            if n % (pf - 1) == 0 and n != 0 and config.real_time_plot is True:

                t0_image = time.perf_counter()

                # plot (processed) image sub-stamps
                stamp1 = imageproc.time_average(stamp1s)
                stamp1 = imageproc.proc(stamp1, dark_stamps[s1], flat_stamps[s1])
                img1.set_data(stamp1.astype(float) ** (pow))
                save_numpy_as_fits(stamp1, os.path.join(out_path, 'stamp1.fits'))

                # autoguiding
                # if temp.fits doesn't exist in share directory, write it
                try:
                    fits.open(os.path.join(config.ag_share_path, 'temp.fits'))
                except FileNotFoundError:
                    ag_stamp = np.mean(autoguide_stamps, axis=0)
                    save_numpy_as_fits(ag_stamp, os.path.join(config.ag_share_path, 'temp.fits'))


                ## reinitialise arrays to hold stamps for real-time plotting
                stamp1s = np.zeros((pf, 2 * ry, 2 * rx))
                stamp_count = 0

                # restore background
                fig.canvas.restore_region(ax1background)

                # redraw just the points
                ax1.draw_artist(img1)

                # fillin the axes rectangle
                fig.canvas.blit(ax1.bbox)

                fig.canvas.flush_events()

                logging.info('Time taken to render figures [ms]: %.3f', 1000 * (time.perf_counter() - t0_image))

            elif n % (pf - 1) == 0 and n != 0 and config.real_time_plot is False:

                # just save (processed) image sub-stamps
                t0_image = time.perf_counter()
                stamp1 = imageproc.time_average(stamp1s)
                stamp1 = imageproc.proc(stamp1, dark_stamps[s1], flat_stamps[s1])
                save_numpy_as_fits(stamp1, os.path.join(out_path, 'stamp1.fits'))
                ## reinitialise arrays to hold stamps for saving time averages
                stamp1s = np.zeros((pf, 2 * ry, 2 * rx))
                stamp_count = 0
                logging.info('Time to save the image stamp [ms]: %.3f', 1000 * (time.perf_counter() - t0_image))

            n += 1

        except ValueError:
            logging.debug('ValueError!!!')
            continue

    ## save results
    #ts = time.perf_counter()
    tbatch = time.perf_counter() - t0_batch
    logging.info('Completed batch %d', batch)
    logging.info('Data rate [Hz]: %d', round(N / tbatch))
    logging.info('Mean image processing time [ms]: %.3f', np.mean(processing_times))

    t0_save = time.perf_counter()
    np.save(os.path.join(out_path, 'photometry_batch%d.npy' % batch), photometry)
    np.save(os.path.join(out_path, 'skys_batch%d.npy' % batch), sky_lvls)
    np.save(os.path.join(out_path, 'times_batch%d.npy' % batch), times)
    np.save(os.path.join(out_path, 'texps_batch%d.npy' % batch), texps)
    np.save(os.path.join(out_path, 'seqs_batch%d.npy' % batch), seqs)
    tsave = time.perf_counter() - t0_save
    logging.info('Time to save batch data [ms]: %.3f', 1000 * tsave)


if ONLINE is True:
    # return camera to normal state
    cam.finish()
    logging.info('Finished live acquistion.')

    logging.info('Closing camera...')
    cam.close()
    logging.info('Camera closed.')

else:
    print('Finished offline test run.')
