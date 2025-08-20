# usage example in terminal
#python run.py 0 0

# load the configuration file
def load_config(path="config.txt"):
    config = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and full-line comments

            # Remove inline comment
            line = line.split("#", 1)[0].strip()

            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip('"').strip("'")  # remove quotes
    return config

config = load_config()

####################################### imports ################################
import os
import sys
import logging

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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')  # No GUI, faster

# suppress numba warning re reflected lists
if config["suppress_numba_warning"] == "True":
    from numba.core.errors import NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

## offline imports
from scipy.ndimage import shift # only used for generating "New" scenes during offline testing
from skimage.io import MultiImage
################################################################################

# useful function to save numpy arrays to fits file format
def save_numpy_as_fits(numpy_array, filename):
    hdu = fits.PrimaryHDU(numpy_array)
    hdul = fits.HDUList([hdu])
    filename_temp = filename + '.part'
    hdul.writeto(filename_temp, overwrite=True)
    os.replace(filename_temp, filename)

# run id and scene id
RUN = int(sys.argv[1])
SCENE = int(sys.argv[2])

# dubugging and housekeeping info written to housekeeping.log
run_dir = os.path.join(config["out_path"], 'RUN_' + str(RUN))
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

###################### initialise logger #######################################
logger = logging.getLogger('housekeeping_logger')
logger.setLevel('INFO')
fh = logging.FileHandler(os.path.join(out_path, "housekeeping.log")) # create file handler which logs
fh.setLevel('INFO')
ch = logging.StreamHandler() # create console handler with the same level
ch.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # create formatter and add it to the handlers
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch) # add the handlers to logger
logger.addHandler(fh)
################################################################################

#################### Online vs offline mode ####################################
## For testing purposes, it is useful to run the software offline on pre-acquired data
## ONLINE = True : Live mode acquistion with Kinetix
## ONLINE = False : (Testing onlyl) Run offline on pre-existing image data
ONLINE = config["online"]
if ONLINE == "True":
    ## Initialise and open the camera
    logger.info('Initialising the camera...')
    pvc.init_pvcam()                   # Initialize PVCAM
    cam = next(Camera.detect_camera()) # Use generator to find first camera.
    cam.open()                         # Open the camera.
    logger.info('Camera open!')

    ## set camera mode
    if config["sensitivity"] == "True":
        cam.readout_port = 0
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('Camera in Sensitivity mode.')
    elif config["speed"] == "True":
        cam.readout_port = 1
        cam.speed_table_index = 0
        cam.gain = 1
        logger.info('Camera set to Speed mode.')
    elif config["dynamic"] == "True":
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
    ref_exp_time = int(config["ref_exp_time"])
    logger.info('Acquring reference frame with a %.2f second integration', (ref_exp_time / 1000))
    ref = cam.get_frame(exp_time=ref_exp_time)
    logger.info('Done!')

else:
    logger.info('Running software offline...')
    ref = fits.getdata(os.path.join(config["offline_path"], config["offline_ref"]))
    ref = ref.astype(np.float32) # change to numpy dtype
    xshift, yshift = 1000, 500 # shifts to generate New scene for testing automated scene change detection
    if SCENE == 1:
        ref = np.flip(ref) # apply rotation
    elif SCENE == 2:
        ref = shift(ref, (xshift, yshift), order=0, cval=np.median(ref)) # apply integer pixel shift
    logger.info('Reference frame loaded.')
################################################################################

## grab calibration frames - N.B. The calibration frames for the Kinetix must be acquired using the same mode of operation
if config["USE_CALIBRATION_FRAMES"] == "True":
    logger.info('Loading calibration frames...')
    flat = np.load(config["flat"])
    dark = np.load(config["dark"])
    logger.info('Found flat and dark frames.')
    logger.info('Flat correcting and dark subtracting reference...')
    ref = imageproc.proc(ref, dark, flat)
    logger.info('Done!')
else:
    logger.info('USE_CALIBRATION_FRAMES is False')

# compute estimates of the noise and sky level
sky, modal_sky_count = imageproc.compute_mode(ref.astype(int))
logger.info('Reference modal sky level: %d [ADU] with %d counts' % (sky, modal_sky_count))
ref -= sky # sky subtract
std = mad_std(ref)
logger.info('Reference MAD [ADU]: %.3f', std)

# save reference for visual inspection
save_numpy_as_fits(ref, os.path.join(fits_path, 'ref.fits'))
logger.info('Saved the reference image to %s' % os.path.join(fits_path, 'ref.fits'))

# run initial peak finding routine on data to detect bright stars
logger.info('Starting source detection routine...')
peaks = find_peaks(ref, threshold=10*std, box_size=int(config["r0"]), border_width=2*int(config["r0"]))

# at least two sources should be detected, otherwise retry reference frame acquisition
if peaks is None or len(peaks) <= 1:
    logger.info('An insufficient number of sources was detected, aborting run to reacquire a reference image.\n')
    cam.close()
    logger.info('Camera closed, run aborted...\n')
    sys.exit()

# sort peaks in descending order by brightness (i.e. the brightest source is first)
peaks.sort('peak_value')
peaks.reverse()

# fit 2D Gaussian to the bright sources to estimate stamp radii in x and y
logger.info('Estimating PSF model and stamp radii')
r = int(config["r0"]) # initial guess for stamp radius
rx, ry, sigma_x, sigma_y, xc_ref, yc_ref, psf_model = imageproc.update_r(ref, peaks, r=r, nsigma=float(config["nsigma"]), lim=int(config["lim"]), img_path=img_path)
logger.info('sigma_x=%.3f, sigma_y=%.3f', sigma_x, sigma_y)
logger.info('rx=%d, ry=%d:', rx, ry)

## heuristically assess the quality of the PSF model
if (rx / ry) >= float(config["sigma_ratio"]) or (ry / rx) >= float(config["sigma_ratio"]):
    logger.info('The fitted PSF model is badly asymmetric... aborting run to reacquire a reference image.\n')
    cam.close()
    logger.info('Camera closed, run aborted...\n')
    sys.exit()

# match-filter the reference image with the PSF Model and normalise to generate a source detection "map"
if config["emp_detect"] == "True":
    D_norm = imageproc.make_detection_map_empirical(ref, psf_model, std)
else:
    D_norm = imageproc.make_detection_map(ref, psf_model, float(config["rdnoise"]), float(config["gain"]))
save_numpy_as_fits(D_norm, os.path.join(fits_path, 'D_norm.fits')) # save for visual inspection

## run a peak finding routine on the normalised detection map
positions = find_peaks(D_norm, threshold=float(config["thresh"]), box_size=int((rx + ry)/2), border_width=(rx + ry))
positions.sort('peak_value')
positions.reverse() # sort so that the highest SNR star is first (Detection map is normalised by uncertainties)
peaks = positions['peak_value'] # flux peaks

########## Source(s) for Autoguiding / Real Time plotting ######################
# compute distance of sources to their closest detected neighbour
min_dist = imageproc.minimum_source_separation(positions)
positions['closest_neighbour_distance [pix]'] = min_dist
nsources = len(positions)
logger.info('Detected sources: %d', nsources)

# s1 will also be used for autoguiding if enabled, and by default ag_dist_thresh is set to 1, but can be increased as needed (e.g. due to bad seeing)
dist_thresh = float(config["ag_dist_thresh"]) * np.sqrt(rx**2 + ry**2) # distance threshold - neighbouring star centroid outside of the aperture?
logger.info('Distance threshold: %.2f' % dist_thresh)
s1, s2, s1_sep, s2_sep = imageproc.find_sources_to_track(positions, dist_thresh)
logger.info('Tracking sources %d and %d' % (s1, s2))
# TODO: WHAT IF NO CANDIDATES FOUND FOR S1 AND S2?
################################################################################

################# Automated background region generation #######################
# KDE of source positions, weighted by brightness, to allow a constrained search in sparsely populated regions
positions = np.vstack((positions['x_peak'], positions['y_peak'])).T # numba doesn't like astropy tables
bb_pos, bb_rs = imageproc.background_boxes(positions, peaks, ref, rx, ry, img_path, bbox_size=int(config["bbox_size"]), N=int(config["nbboxes"]))
nboxes = len(bb_pos)
if nboxes == 0:
    logger.info('No valid background boxes found... aborting run to reacquire a new reference image.\n')
    cam.close()
    logger.info('Camera closed, run aborted...\n')
    sys.exit()
logger.info('Number of background region boxes: %d', nboxes)
################################################################################

####### Plots, housekeeping data and JIT compilation ###########################
# plot detection map and annotated reference image
imageproc.plot_detection_map(D_norm, positions, os.path.join(img_path, 'D_norm.png'))
imageproc.plot_annotated_reference(ref, sky, positions, rx, ry, bb_pos, bb_rs, os.path.join(img_path, 'ref_annotated.png'))

# save source and sky background box positions
np.save(os.path.join(phot_path, 'positions.npy'), positions)
np.save(os.path.join(phot_path, 'bb_pos.npy'), bb_pos)
np.save(os.path.join(phot_path, 'bb_rs.npy'), bb_rs)

# generate calibration frame stamps for each source aperture
if config["USE_CALIBRATION_FRAMES"] == "True":
    logger.info('Generating calibration frame stamps for the source apertures and sky background boxes...')
    dark_stamps, flat_stamps = imageproc.calibration_stamps(np.copy(dark), np.copy(flat), positions, rx, ry)
    sky_dark_stamps, sky_flat_stamps = imageproc.sky_calibration_stamps(np.copy(dark), np.copy(flat), bb_pos, bb_rs)
    logger.info('Done!')

logger.info('JIT compiling functions...')
if config["USE_CALIBRATION_FRAMES"] == "True":
    phot = imageproc.fluxes_stamps(ref, dark_stamps, flat_stamps, positions, nsources, rx, ry)
    skys = imageproc.sky_stamps(ref, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
else:
    phot = imageproc.fluxes_stamps_nocal(ref, positions, nsources, rx, ry)
    skys = imageproc.sky_stamps_nocal(ref, nboxes, bb_pos, bb_rs)
logger.info('Done!')

pf = int(config["plot_freq"]) # plot / stamp save frequency

# initialise real-time plotting
if config["real_time_plot"] == "True":
    pow = float(config["pow"]) # power-law rescaling of stamp data for better viewing
################################################################################

######################### Data acquistion ######################################
if ONLINE == "True":
    ## setup a live mode acquistion - frames stored in buffer
    logger.info('Starting live mode acquistion...')
    exp_time = float(config["exp_time"]) # ms
    logger.info('Exposure time [ms]: %f', exp_time)
    cam.start_live(exp_time=exp_time, buffer_frame_count=int(config["buffer_count"]), stream_to_disk_path=None)
    logger.info('Camera now collecting data...')
else:
    ## offline mode - run on pre-acquired imaging data
    path = config["offline_path"]
    ordered_files = imageproc.find_offline_mode_data(path)

# initialise scene change criteria
NEW_SCENE = False
med_stamp1_fluxes, med_stamp2_fluxes = [], [] # list of historical stamp median fluxes
candidate_change = [] # candidate change events
change_counter = 0 # initialise change_counter (checks for consecutive signficant deviations from historical median flux)

# initialise performance counter
t0 = time.perf_counter()

# number of data batches: total number of images processed = batches * N
batches = int(config["batches"])
for batch in range(batches):
    # if runnning software offline, load the test imagery as if being acquired by camera during a live acquisition
    if ONLINE == "False":
        file_name = os.path.join(path, ordered_files[batch])
        logger.info('Loading data from %s' % file_name)
        img_coll = np.array(MultiImage(file_name))[0]

    # time batch completion
    t0_batch = time.perf_counter()

    # generate arrays to store photometry and background measurements
    N = int(config["N"]) # total number of exposures to acquire per batch
    photometry, sky_lvls = np.zeros((N, len(positions))), np.zeros((N, len(bb_pos)))

    # generate arrays to store diagnostic/housekeeping data
    times, texps, seqs, processing_times = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    # initialise the arrays to hold the stamps for real-time plotting and automated scene change detection
    stamp1s, stamp2s, stamp_count = np.zeros((pf, 2 * ry, 2 * rx)), np.zeros((pf, 2 * ry, 2 * rx)), 0

    # initialise the array for holding the full image data
    ref_array = np.zeros(ref.shape, dtype=np.uint16)

    n = 0 # counter for number of acquired frames
    while n < N:
        try:
            if ONLINE == "True":
                frame = cam.poll_frame(copyData=False) # oldest frame in buffer popped from the queue
                times[n] = time.perf_counter() - t0 # store time exposure acquired
                texps[n] = frame[1] # exptime (housekeeping)
                seqs[n] = frame[2] # sequence number (housekeeping)
                data = frame[0]['pixel_data'] # read the data
            else:
                data = img_coll[n] # read the data (offline)
                seqs[n] = n
                # automated scene change detection tests: apply rotationn or integer shifts to create a "New" scene
                if SCENE == 0 and batch >= 6:
                    data = np.random.normal(100, 10, data.shape).astype(np.uint16)
                elif SCENE == 1 and batch >= 6:
                    data = shift(data, (xshift, yshift), order=0, cval=np.median(data)).astype(np.uint16)
                elif SCENE == 1:
                    data = np.flip(data)
                #elif SCENE == 2:
                #    data = shift(data, (xshift, yshift), order=0, cval=np.median(data)).astype(np.uint16)

            # initialise a performance counter for the image processing ####
            tproc = time.perf_counter()

            # aperture photometry and sky region levels
            if config["USE_CALIBRATION_FRAMES"] == 'True':
                phot = imageproc.fluxes_stamps(data, dark_stamps, flat_stamps, positions, nsources, rx, ry)
                skys = imageproc.sky_stamps(data, sky_dark_stamps, sky_flat_stamps, nboxes, bb_pos, bb_rs)
            else:
                phot = imageproc.fluxes_stamps_nocal(data, positions, nsources, rx, ry)
                skys = imageproc.sky_stamps_nocal(data, nboxes, bb_pos, bb_rs)

            # store aperture photometry and sky background estimates
            photometry[n], sky_lvls[n] = phot, skys

            # compute processing time [ms]
            processing_times[n] = 1000 * (time.perf_counter() - tproc)

            # cache source stamps
            stamp1s[stamp_count] = data[positions[s1][1] - ry : positions[s1][1] + ry, positions[s1][0] - rx : positions[s1][0] + rx]
            stamp2s[stamp_count] = data[positions[s2][1] - ry : positions[s2][1] + ry, positions[s2][0] - rx : positions[s2][0] + rx]
            stamp_count += 1

            # add data to the reference image array
            ref_array += data

            ## hold processing for real time plotting and automated scene change detection
            if n % (pf - 1) == 0 and n != 0 and config["real_time_plot"] == "True":
                t0_scene_change = time.perf_counter() # check the scene change overhead
                # process the tracked source stamps
                if config["USE_CALIBRATION_FRAMES"] == "True":
                    stamp1, stamp2 = imageproc.proc(imageproc.time_average(stamp1s), dark_stamps[s1], flat_stamps[s1]), imageproc.proc(imageproc.time_average(stamp2s), dark_stamps[s2], flat_stamps[s2])
                else:
                    stamp1, stamp2 = imageproc.time_average(stamp1s), imageproc.time_average(stamp2s)

                ## compare stamp_flux to historial stamp fluxes ##
                stamp1_flux, stamp2_flux, = np.sum(stamp1), np.sum(stamp2) # time-averaged stamp flux
                stamp1_flux_hist, stamp2_flux_hist = np.median(med_stamp1_fluxes), np.median(med_stamp2_fluxes) # historical median
                stamp1_flux_std, stamp2_flux_std = mad_std(med_stamp1_fluxes), mad_std(med_stamp2_fluxes) # historicdal MAD scaled to std

                # check aperture stamp has not shifted badly in time since the reference exposure frame was acquired
                if len(med_stamp1_fluxes) <= int(config["burn_in"]):
                    s1_off_centre, s1_logger_info = imageproc.check_stamp_centrality(s1, stamp1, float(config["stamp_centrality_thresh"]))
                    s2_off_centre, s2_logger_info = imageproc.check_stamp_centrality(s2, stamp2, float(config["stamp_centrality_thresh"]))
                    logger.info(s1_logger_info[0])
                    logger.info(s2_logger_info[0])
                    if s1_off_centre == True or s2_off_centre == True:
                        logger.info('Source stamps are badly off-centre, restarting reference frame acquisition.\n')
                        cam.finish() # return camera to normal state...
                        cam.close() # ...and close
                        logger.info('Camera closed, run aborted...\n')
                        sys.exit()

                # check if the scene has changed
                candidate_change, change_counter, NEW_SCENE, logger_info = imageproc.scene_change_check([s1, s2],
                                            [med_stamp1_fluxes, med_stamp2_fluxes], [stamp1_flux, stamp2_flux],
                                            [stamp1_flux_hist, stamp2_flux_hist], [stamp1_flux_std, stamp2_flux_std],
                                            rx, ry, int(config["burn_in"]), sky_lvls, change_counter, candidate_change,
                                            float(config["scene_change_sky_thresh"]), float(config["scene_change_flux_thresh"]), int(config["consecutive"]), NEW_SCENE)
                if len(logger_info) > 0:
                    for info in logger_info:
                        logger.info(info)

                # add med to list of historical med fluxes
                med_stamp1_fluxes.append(stamp1_flux)
                med_stamp2_fluxes.append(stamp2_flux)

                # save the source stamps as a .fits files
                save_numpy_as_fits(stamp1, os.path.join(fits_path, 'source_%d.fits' % s1))
                save_numpy_as_fits(stamp2, os.path.join(fits_path, 'source_%d.fits' % s2))
                save_numpy_as_fits(ref_array, os.path.join(fits_path, 'ref_temp.fits'))

                ## reinitialise arrays to hold stamps for real-time plotting and scene change decisions
                stamp1s, stamp2s = np.zeros((pf, 2 * ry, 2 * rx)), np.zeros((pf, 2 * ry, 2 * rx))
                stamp_count = 0 # reset stamp count

                # reinitialise the array for holding the iamge data
                ref_array = np.zeros(ref.shape, dtype=np.uint16)

            # update the image no. counter
            n += 1

        except Exception as e:
            print(e)
            logging.debug(e)
            continue

        # if the scene has changed, kill the script
        if NEW_SCENE == True:
            logger.info('The scene has changed, aborting run.\n')
            if ONLINE == "True":
                # return camera to normal state and close
                cam.finish()
                cam.close()
                logger.info('Camera closed, run aborted...\n')
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

if ONLINE == "True":
    # return camera to normal state
    cam.finish()
    logger.info('Finished live acquistion. Closing the camera...')
    cam.close()
    logger.info('Camera closed.')
else:
    logger.info('Finished offline test run.')
