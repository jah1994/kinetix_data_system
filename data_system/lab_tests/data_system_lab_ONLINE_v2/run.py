################# imports ########################
import config # software configuration file
import os
import sys
import logging

# usage example in terminal
#python run.py 0 0

# run id and scene id
RUN = int(sys.argv[1])
SCENE = int(sys.argv[2])


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

# only used for generating "New" scenes during offline testing
from scipy.ndimage import shift

# suppress numba warning re reflected lists
if config.suppress_numba_warning is True:
    from numba.core.errors import NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# useful function to save numpy arrays to fits file format
def save_numpy_as_fits(numpy_array, filename):
    hdu = fits.PrimaryHDU(numpy_array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)

# dubugging and housekeeping info written to housekeeping.log
#out_path = config.out_path + '/RUN_' + str(RUN) + '/SCENE_' + str(SCENE)
run_dir = os.path.join(config.out_path, 'RUN_' + str(RUN))
out_path = os.path.join(run_dir, 'SCENE_' + str(SCENE))
phot_path = os.path.join(out_path, 'photometry')
fits_path = os.path.join(out_path, 'fits')
img_path = os.path.join(out_path, 'images')

# if run directory doesn't exist, make it
if os.path.exists(run_dir) == False:
    print('Making directory for RUN %d' % RUN)
    os.mkdir(run_dir)

# if out_path doesn't exist, make it
if os.path.exists(out_path) == False:
    print('Making result directory:', out_path)
    os.mkdir(out_path)
    os.mkdir(phot_path)
    os.mkdir(fits_path)
    os.mkdir(img_path)
else:
    print('Result directory already exists:', out_path)

### initialise logger ###
logger = logging.getLogger('housekeeping_logger')
logger.setLevel('INFO')
# create file handler which logs
fh = logging.FileHandler(os.path.join(out_path, "housekeeping.log"))
fh.setLevel('INFO')
# create console handler with the same level
ch = logging.StreamHandler()
ch.setLevel('INFO')
# create formatter and add it to the handlers
#formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

### Online vs offline mode ####
## For testing purposes, it is useful to run the software offline on pre-acquired data
## ONLINE = True : Live mode acquistion with Kinetix
## ONLINE = False : (Testing onlyl) Run offline on pre-acquired imaging data
ONLINE = config.online

## grab calibration frames - N.B. The calibration frames for the Kinetix must be acquired using the same mode of operation
if config.USE_CALIBRATION_FRAMES is True:
    logger.info('Loading calibration frames...')
    flat = np.load(config.flat)
    dark = np.load(config.dark)
    logger.info('Found flat and dark frames.')
else:
    logger.info('USE_CALIBRATION_FRAMES is False')

if ONLINE is True:

    ## Initialise and open the camera
    logger.info('Initialising the camera...')
    pvc.init_pvcam()                   # Initialize PVCAM
    cam = next(Camera.detect_camera()) # Use generator to find first camera.
    cam.open()                         # Open the camera.
    logger.info('Camera open!')

    ## set camera mode
    if config.sensitivity is True:
        cam.readout_port = 0
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('Camera in Sensitivity mode.')
    elif config.speed is True:
        cam.readout_port = 1
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('Camera set to Speed mode.')
    elif config.dynamic is True:
        cam.readout_port = 2
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('Camera in Dynamic Range mode.')
    else:
        cam.readout_port = 0
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('No camera mode specified, defaulting to Sensitivity mode...')


    ### Acquire reference frame - return source positions and aperture size
    ref_exp_time = config.ref_exp_time
    logger.info('Acquring reference frame with a %.2f second integration', (ref_exp_time / 1000))
    ref = cam.get_frame(exp_time=ref_exp_time)
    logger.info('Done!')

else:
    logger.info('Running software offline...')
    ref = fits.getdata(os.path.join(config.offline_path, config.offline_ref))
    ref = ref.astype(np.float32) # change to numpy dtype

    # shifts to generate New scene for testing automated scene change detection
    xshift, yshift = 1000, 500

    # apply rotation
    if SCENE == 1:
        ref = np.flip(ref)
    # apply integer pixel shift
    elif SCENE == 2:
        ref = shift(ref, (xshift, yshift), order=0, cval=np.median(ref))
    logger.info('Reference frame loaded.')


if config.USE_CALIBRATION_FRAMES is True:
    logger.info('Flat correcting and dark subtracting reference...')
    ref = imageproc.proc(ref, dark, flat)
    logger.info('Done!')

# compute estimates of the noise and sky level
#sky = np.median(ref)
#logger.info('Reference sky level [ADU]: %.3f', sky)
sky, modal_sky_count = imageproc.compute_mode(ref.astype(int))
logger.info('Reference modal sky level: %d [ADU] with %d counts' % (sky, modal_sky_count))
ref -= sky # sky subtract
std = mad_std(ref)
logger.info('Reference MAD [ADU]: %.3f', std)

# save reference for visual inspection
save_numpy_as_fits(ref, os.path.join(fits_path, 'ref.fits'))
logger.info('Saved the reference image as a ref.fits file.')

# run initial peak finding routine on data to detect bright stars
logger.info('Starting source detection routine...')
peaks = find_peaks(ref, threshold=10*std, box_size=config.r0, border_width=2*config.r0)

# at least two sources should be detected, otherwise retry reference frame acquisition
if peaks is None or len(peaks) <= 1:
    logger.info('An insufficient number of sources was detected, aborting run to reacquire a reference image.\n')
    sys.exit()

# sort peaks so that the brightest source is the first
peaks.sort('peak_value')
peaks.reverse()

# fit 2D Gaussian to the bright sources to estimate stamp radii in x and y
logger.info('Estimating PSF model and stamp radii')
r = config.r0 # initial guess for stamp radius
rx, ry, sigma_x, sigma_y, psf_model = imageproc.update_r(ref, peaks, r=r, nsigma=config.nsigma, lim=config.lim, img_path=img_path)
logger.info('sigma_x=%.3f, sigma_y=%.3f', sigma_x, sigma_y)
logger.info('rx=%d, ry=%d:', rx, ry)

## heuristically assess the quality of the PSF model
# test 1) (x,y) asymmetry
if (rx / ry) >= 2 or (ry / rx) >= 2:
    logger.info('The fitted PSF model is badly asymmetric... aborting run to reacquire a reference image.\n')
    sys.exit()

# match-filter reference with the PSF Model and normalise to generate a detection map
if config.emp_detect is True:
    D_norm = imageproc.make_detection_map_empirical(ref, psf_model, std)
else:
    D_norm = imageproc.make_detection_map(ref, psf_model, config.rdnoise, config.gain)
save_numpy_as_fits(D_norm, os.path.join(fits_path, 'D_norm.fits')) # save for visual inspection

## run a peak finding routine on the normalised detection map
positions = find_peaks(D_norm, threshold=config.thresh, box_size=int((rx + ry)/2), border_width=(rx + ry))
positions.sort('peak_value')
positions.reverse() # sort so that the highest SNR star is first (Detection map is normalised by uncertainties)
peaks = positions['peak_value'] # flux peaks

####### Real Time plotting / Saving source stamps###
# find sufficiently isolated bright star
min_dist = []
for i,pos in enumerate(positions):
    xci, yci = pos['x_peak'], pos['y_peak']
    dists = []
    for j,pos in enumerate(positions):
        if i != j:
            xcj, ycj = pos['x_peak'], pos['y_peak']
            dists.append(np.sqrt((xci - xcj)**2 + (yci - ycj)**2))
    min_dist.append(min(dists))
positions['closest_neighbour_distance [pix]'] = min_dist

# distance threshold - neighbouring star centroid outside of the aperture?
dist_thresh = np.sqrt(rx**2 + ry**2)
logger.info('Distance threshold: %.2f' % dist_thresh)
S1, S2 = False, False
for i,pos in enumerate(positions):
    if pos['closest_neighbour_distance [pix]'] >= dist_thresh and S1 is False:
        s1, S1 = i, True
        logger.info('Tracking source %d' % s1)
        logger.info('Closest neighbour is %d pixels away' % int(pos['closest_neighbour_distance [pix]']))
    elif pos['closest_neighbour_distance [pix]'] >= dist_thresh and S1 is True:
        s2, S2 = i, True
        logger.info('Tracking source %d' % s2)
        logger.info('Closest neighbour is %d pixels away' % int(pos['closest_neighbour_distance [pix]']))
        break
###############################################
positions = np.vstack((positions['x_peak'], positions['y_peak'])).T # numba doesn't like astropy tables
nsources = len(positions)
logger.info('Detected sources: %d', nsources)

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
plt.savefig(os.path.join(img_path, 'D_norm.png'), bbox_inches='tight') # save to disk for reference
#plt.show();
plt.close();

### automated background region selection
# KDE of source positions, weighted by brightness, to allow a constrained search in sparsely populated regions
bb_pos, bb_rs = imageproc.background_boxes(positions, peaks, ref, rx, ry, img_path, bbox_size=config.bbox_size, N=config.nbboxes)
nboxes = len(bb_pos)
if nboxes == 0:
    logger.info('No valid background boxes found... aborting run to reacquire a new reference image.\n')
    sys.exit()
logger.info('Number of background region boxes: %d', nboxes)

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
for p,pos in enumerate(positions):
    ax.add_patch(patches.Rectangle(xy=(pos[0]-rx, pos[1]-ry),
                                   width=2*rx, height=2*ry, fill=False,
                                   label=p))
    ax.text(pos[0], pos[1], str(p), fontsize=fs)

for j, (pos, rs) in enumerate(zip(bb_pos, bb_rs)):

    ax.add_patch(patches.Rectangle(xy=(pos[0] - rs[0], pos[1] - rs[1]),
                                   width=2*rs[0], height=2*rs[1], fill=False, label=j, color='red'))
    ax.text(pos[0], pos[1], str(j), fontsize=fs, c='r')
plt.savefig(os.path.join(img_path, 'ref_annotated.png'), bbox_inches='tight')
plt.close();

#### Housekeeping data ###
np.save(os.path.join(phot_path, 'positions.npy'), positions) # save positions of sources
np.save(os.path.join(phot_path, 'bb_pos.npy'), bb_pos) # save backbround box positions...
np.save(os.path.join(phot_path, 'bb_rs.npy'), bb_rs) #... and their raddii

# generate calibration frame stamp_size
if config.USE_CALIBRATION_FRAMES is True:
    logger.info('Generating calibration frame stamps for the source apertures and sky background boxes...')
    dark_stamps, flat_stamps = imageproc.calibration_stamps(np.copy(dark), np.copy(flat), positions, rx, ry)
    sky_dark_stamps, sky_flat_stamps = imageproc.sky_calibration_stamps(np.copy(dark), np.copy(flat), bb_pos, bb_rs)
    logger.info('Done!')

############ Live mode acquistion #################
logger.info('JIT compiling functions...')
if config.USE_CALIBRATION_FRAMES is True:
    phot = imageproc.fluxes_stamps(ref, dark_stamps, flat_stamps, positions, nsources, rx, ry)
    skys = imageproc.sky_stamps(ref, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
else:
    phot = imageproc.fluxes_stamps_nocal(ref, positions, nsources, rx, ry)
    skys = imageproc.sky_stamps_nocal(ref, nboxes, bb_pos, bb_rs)
logger.info('Done!')

pf = config.plot_freq # plot frequency / stamp save frequency

# initialise real-time plot
if config.real_time_plot is True:

    fig = plt.figure(figsize=(2, 2))

    # power-law rescaling of stamp data
    pow = config.pow

    ax1 = fig.add_subplot(1, 1, 1) # image

    RX, RY = np.linspace(0, 2 * rx, num= 2 * rx), np.linspace(0, 2* ry, num= 2 * ry)
    X, Y = np.meshgrid(RX, RY)

    if config.USE_CALIBRATION_FRAMES is True:
        minview = int(np.median(dark))
        maxview = minview + 100
    else:
        minview = sky ** pow
        maxview = minview + 1000
    img1 = ax1.imshow(X, cmap="Greys", vmin=minview, vmax=maxview, origin='lower')
    ax1.set_title('Source: %d' % s1)

    fig.canvas.draw()

    # cache the background (blitting)
    ax1background = fig.canvas.copy_from_bbox(ax1.bbox)

    plt.show(block=False)


if ONLINE is True:
    ## setup a live mode acquistion - frames stored in buffer
    logger.info('Starting live mode acquistion...')
    exp_time = config.exp_time # ms
    logger.info('Exposure time [ms]: %f', exp_time)
    cam.start_live(exp_time=exp_time, buffer_frame_count=config.buffer_count, stream_to_disk_path=None)
    logger.info('Camera now collecting data...')

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

### scene change automation ####
NEW_SCENE = False
med_stamp1_fluxes, med_stamp2_fluxes = [], [] # list of historical stamp median fluxes
candidate_change = [] # candidate change events
change_counter = 0 # initialise change_counter (checks for consecutive signficant deviations from historical median flux)
for batch in range(batches):

    # if runnning software offline, load the test imagery as if being acquired by camera during a live acquisition
    if ONLINE is False:
        file_name = os.path.join(path, ordered_files[batch])
        logger.info('Loading data from %s' % file_name)
        img_coll = np.array(MultiImage(file_name))[0]
        print(img_coll.shape, img_coll.dtype)
        logger.info('Done!')

    # time batch completion
    t0_batch = time.perf_counter()

    N = config.N # total number of exposures to acquire per batch

    # generate arrays to store photometry and background measurements
    photometry = np.zeros((N, len(positions)))
    sky_lvls = np.zeros((N, len(bb_pos)))

    # generate arrays to store diagnostic/housekeeping data
    times, texps, seqs, processing_times = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    ## initialise the arrays to hold the stamps for real-time plotting and automated scene change detection
    stamp1s, stamp2s = np.zeros((pf, 2 * ry, 2 * rx)), np.zeros((pf, 2 * ry, 2 * rx))
    stamp_count = 0

    ## initialise array for autoguiding stamp
    if config.autoguide is True:
        autoguide_stamps = np.zeros((pf, config.ag_stamp_size, config.ag_stamp_size))
        ag_r = int(config.ag_stamp_size / 2)

    n = 0 # counter for number of acquired frames
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
                # pull the data (offline)
                data = img_coll[n]

                # automated scene change detection tests: apply rotationn or integer shifts to create a "New" scene
                if SCENE == 0 and batch >= 6:
                    #data = np.flip(data)
                    data = np.random.normal(0,1, data.shape)
                elif SCENE == 1 and batch >= 6:
                    data = shift(data, (xshift, yshift), order=0, cval=np.median(data))
                elif SCENE == 1:
                    data = np.flip(data)
                elif SCENE == 2:
                    data = shift(data, (xshift, yshift), order=0, cval=np.median(data))

            ### Time the workhorse ####
            tproc = time.perf_counter()

            # aperture photometry and sky region levels
            if config.USE_CALIBRATION_FRAMES is True:
                phot = imageproc.fluxes_stamps(data, dark_stamps, flat_stamps, positions, nsources, rx, ry)
                skys = imageproc.sky_stamps(data, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
            else:
                phot = imageproc.fluxes_stamps_nocal(data, positions, nsources, rx, ry)
                skys = imageproc.sky_stamps_nocal(data, nboxes, bb_pos, bb_rs)

            # store aperture photometry and sky background estimates
            photometry[n] = phot
            sky_lvls[n] = skys

            # compute processing time [ms]
            processing_times[n] = 1000 * (time.perf_counter() - tproc)

            ## cache source stamps
            stamp1s[stamp_count] = data[positions[s1][1] - ry : positions[s1][1] + ry, positions[s1][0] - rx : positions[s1][0] + rx]
            stamp2s[stamp_count] = data[positions[s2][1] - ry : positions[s2][1] + ry, positions[s2][0] - rx : positions[s2][0] + rx]

            # cache autoguide stamp
            if config.autoguide is True:
                autoguide_stamps[stamp_count] = data[positions[s1][1] - ag_r : positions[s1][1] + ag_r, positions[s1][0] - ag_r : positions[s1][0] + ag_r]

            stamp_count += 1

            ## hold processing for real time plotting and auotmated scene change detection
            if n % (pf - 1) == 0 and n != 0 and config.real_time_plot is True:

                # check plotting overhead
                t0_image = time.perf_counter()

                # process the tracked source stamps
                if config.USE_CALIBRATION_FRAMES is True:
                    stamp1, stamp2 = imageproc.proc(imageproc.time_average(stamp1s), dark_stamps[s1], flat_stamps[s1]), imageproc.proc(imageproc.time_average(stamp2s), dark_stamps[s2], flat_stamps[s2])
                else:
                    stamp1, stamp2 = imageproc.time_average(stamp1s), imageproc.time_average(stamp2s)

                ## compare stamp_flux to historial stamp fluxes ##
                stamp1_flux, stamp2_flux, = np.sum(stamp1), np.sum(stamp2) # time-averaged stamp flux
                stamp1_flux_hist, stamp2_flux_hist = np.median(med_stamp1_fluxes), np.median(med_stamp2_fluxes) # historical median
                stamp1_flux_std, stamp2_flux_std = mad_std(med_stamp1_fluxes), mad_std(med_stamp2_fluxes) # historicdal MAD scaled to std

                ### SCENE CHANGE CRITERIA ###
                if len(med_stamp1_fluxes) > config.burn_in and len(med_stamp2_fluxes) > config.burn_in:
                    # Crietrion 1: Is the measured flux of the two tracked sources comparable to the measured background flux
                    ap_sky_lvl = ((2 * rx) * (2 * ry)) * (np.median(sky_lvls[sky_lvls != 0]))
                    if abs(stamp1_flux / ap_sky_lvl) < config.scene_change_sky_thresh and abs(stamp2_flux / ap_sky_lvl) < config.scene_change_sky_thresh:
                        candidate_change.append(change_counter)
                        logger.info('Estimated source aperture sky flux: %.2f' % ap_sky_lvl)
                        logger.info('Estimated flux of source %d: %.2f' % (s1, stamp1_flux))
                        logger.info('Estimated flux of source %d: %.2f' % (s2, stamp2_flux))
                        logger.info('The ratio of the measured flux of source %d and the estimated sky flux of the aperture is %.2f' % (s1, stamp1_flux / ap_sky_lvl))
                        logger.info('The ratio of the measured flux of source %d and the estimated sky flux of the aperture is %.2f' % (s2, stamp2_flux / ap_sky_lvl))
                    # Criterion 2: Are there signficant changes in measured flux for the two tracked sources?
                    elif stamp1_flux < (stamp1_flux_hist - config.scene_change_flux_thresh * stamp1_flux_std) or stamp1_flux > (stamp1_flux_hist + config.scene_change_flux_thresh * stamp1_flux_std):
                        if stamp2_flux < (stamp2_flux_hist - config.scene_change_flux_thresh * stamp2_flux_std) or stamp2_flux > (stamp2_flux_hist + config.scene_change_flux_thresh * stamp2_flux_std):
                            candidate_change.append(change_counter)
                            stamp1_sigma, stamp2_sigma = (stamp1_flux - stamp1_flux_hist) / stamp1_flux_std, (stamp2_flux - stamp2_flux_hist) / stamp2_flux_std
                            logger.info('Significant deviations from baseline fluxes for sources %d and %d' % (s1, s2))
                            logger.info('Measured brightness of source %d has changed by %.2f sigmas' % (s1, stamp1_sigma))
                            logger.info('Measured brightness of source %d has changed by %.2f sigmas' % (s2, stamp2_sigma))

                    # check for consecutive change flags; only if there's consistently flagged change do we abort the run
                    if len(candidate_change) >= config.consecutive and ((candidate_change[-1] - candidate_change[-config.consecutive]) == config.consecutive - 1) == True:
                        NEW_SCENE = True

                # update change_counter
                change_counter += 1

                # add med to list of historical med fluxes
                med_stamp1_fluxes.append(stamp1_flux)
                med_stamp2_fluxes.append(stamp2_flux)

                # save the plotted source stamp as a stamp.fits file
                save_numpy_as_fits(stamp1, os.path.join(fits_path, 'source_%d.fits' % s1))

                ## autoguiding
                if config.autoguide is True:
                    # if temp.fits doesn't exist in share directory, write it
                    try:
                        fits.open(os.path.join(config.ag_share_path, 'temp.fits'))
                    except FileNotFoundError:
                        ag_stamp = np.mean(autoguide_stamps, axis=0)
                        save_numpy_as_fits(ag_stamp, os.path.join(config.ag_share_path, 'temp.fits'))

                ## reinitialise arrays to hold stamps for real-time plotting and scene change decisions
                stamp1s, stamp2s = np.zeros((pf, 2 * ry, 2 * rx)), np.zeros((pf, 2 * ry, 2 * rx))
                stamp_count = 0 # reset stamp count

                ## plot the source stamp (auto scaled for better viewing)
                img1.set_data(stamp1.astype(float) ** (pow))
                fig.canvas.restore_region(ax1background) # restore background
                ax1.draw_artist(img1) # redraw just the updated data
                fig.canvas.blit(ax1.bbox) # fillin the axes rectangle
                fig.canvas.flush_events()
                logger.info('Time taken to render figures [ms]: %.3f', 1000 * (time.perf_counter() - t0_image))

            # update the image no. counter
            n += 1

        except Exception as e:
            print(e)
            logging.debug(e)
            continue

        # if the scene has changed, kill the script
        if NEW_SCENE == True:
            logger.info('The scene has changed, aborting run.\n')
            sys.exit()

    ## save results
    tbatch = time.perf_counter() - t0_batch
    logger.info('Completed batch %d', batch)
    logger.info('Data rate [Hz]: %d', round(N / tbatch))
    logger.info('Mean image processing time [ms]: %.3f', np.mean(processing_times))

    t0_save = time.perf_counter()
    np.save(os.path.join(phot_path, 'photometry_batch%d.npy' % batch), photometry)
    np.save(os.path.join(phot_path, 'skys_batch%d.npy' % batch), sky_lvls)
    np.save(os.path.join(phot_path, 'times_batch%d.npy' % batch), times)
    np.save(os.path.join(phot_path, 'texps_batch%d.npy' % batch), texps)
    np.save(os.path.join(phot_path, 'seqs_batch%d.npy' % batch), seqs)
    logger.info('Time to save batch data [ms]: %.3f', 1000 * (time.perf_counter() - t0_save))

if ONLINE is True:
    # return camera to normal state
    cam.finish()
    logger.info('Finished live acquistion. Closing the camera...')
    cam.close()
    logger.info('Camera closed.')
else:
    logger.info('Finished offline test run.')

'''
elif n % (pf - 1) == 0 and n != 0 and config.real_time_plot is False:

    # just save (processed) image sub-stamps
    t0_image = time.perf_counter()
    if config.USE_CALIBRATION_FRAMES is True:
        stamp1 = imageproc.proc(imageproc.time_average(stamp1s), dark_stamps[s1], flat_stamps[s1])
    else:
        stamp1 = imageproc.time_average(stamp1s)
    save_numpy_as_fits(stamp1, os.path.join(fits_path, 'stamp1.fits'))

    ## reinitialise arrays to hold stamps for saving time averages
    stamp1s, stamp2s = np.zeros((pf, 2 * ry, 2 * rx)), np.zeros((pf, 2 * ry, 2 * rx))
    stamp_count = 0
    logger.info('Time to save the image stamp [ms]: %.3f', 1000 * (time.perf_counter() - t0_image))
'''
