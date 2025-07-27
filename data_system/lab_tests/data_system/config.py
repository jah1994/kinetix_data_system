# run software in online or offline ### Online vs offline mode ####
## For testing purposes, it is useful to run the software offline on pre-acquired data
## ONLINE = True : Live mode acquistion with Kinetix
## ONLINE = False : (Testing onlyl) Run offline on pre-acquired imaging data
online = True

## OFFLINE only ##
offline_path = "data" # path to test data in offline mode only
offline_ref = "M2.fits" # reference image in the offline_path directory to be used in offline mode

# file out path
out_path = "results_NGC884_Sept21_speed3"

# Kinetix mode
sensitivity = False
dynamic = False
speed = True


## ONLINE only ##
ref_exp_time = 1000 # [ms] the exposure time for capturing the reference image
exp_time = 3 # [ms] live feed acquisition mode exposure time
buffer_count = 100 # [images] the maximum number of frames allowed in the circular FIFO buffer
###################

batches = 300 # number of data batches: total number of images processed = batches * N
N = 10000 # total number of exposures to acquire per batch

# calibration file paths
flat = "data/master_flat_test_speed_improved.npy"
dark = "data/master_bias_3ms_speed_take2.npy"

r0 = 60 # initial guess at stamp radii (used to specify stamp size to fit PSF model)
nsigma = 1.5 # apeture stamp radii will be nsigma * sigma
lim = 5 # number of bright stars to use for estimating the PSF model

# source detection
rdnoise = 1.2 # e- Sensitivity mode: 1.2, Dynamic mode: 1.6, Speed mode: 2
gain = 0.25 # e-/ADU Sensitivity mode: 0.25, Dynamic mode: 0.23, Speed mode: 0.85
thresh = 100 # threshold (in sigmas) of detection map
emp_detect = True # use an empirical estimate of data noise in place of noise model (overides rdnoise and gain)

real_time_plot = True # toggle whether to plot output in real time
plot_freq = 2500 # [images] how frequent to update the real-time plotting / save the stamp data
pow = 2 # power-law scaling of real time plotted stamps

ag_stamp_size = 80
ag_share_path = 'C:/Users/rgomer/Desktop/share'

suppress_numba_warning = True
