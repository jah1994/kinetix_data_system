# run software in online or offline ### Online vs offline mode ####
## For testing purposes, it is useful to run the software offline on pre-acquired data
## ONLINE = True : Live mode acquistion with Kinetix
## ONLINE = False : (Testing onlyl) Run offline on pre-acquired imaging data
online = False

## OFFLINE only ##
offline_path = "D:/offline_tests/data" # path to test data in offline mode only
offline_ref = "M2.fits" # reference image in the offline_path directory to be used in offline mode

# file out path
out_path = "D:/offline_tests/" # "C:/Users/rgomer/data_system_lab_ONLINE_v2/Results/Results_19th_Feb/"

# Kinetix mode
sensitivity = True
dynamic = False
speed = False

##### ONLINE only ####
ref_exp_time = 5000 # [ms] the exposure time for capturing the reference image
exp_time = 12 # [ms] live feed acquisition mode exposure time
buffer_count = 100 # [images] the maximum number of frames allowed in the circular FIFO buffer
###################

batches = 9 # 10000 (Speed mode) 1000 (Sensitivity mode) # number of data batches: total number of images processed = batches * N
N = 104 # 2000 (Sensitivity mode) 500 (Sensitivity mode) # total number of exposures to acquire per batch

# calibration file paths
USE_CALIBRATION_FRAMES = True
flat = "D:/offline_tests/data/master_flat_3200.npy" # "C:/Users/rgomer/data_system_lab_ONLINE_v2/calibration_images/master_flat_Sensitivity.npy"
dark = "D:/offline_tests/data/master_dark_3200.npy" # "C:/Users/rgomer/data_system_lab_ONLINE_v2/calibration_images/master_dark_Sensitivity_12ms.npy"

r0 = 30 # initial guess at stamp radii (used to specify stamp size to fit PSF model)
nsigma = 2.5 # apeture stamp radii will be nsigma * sigma
lim = 5 # number of bright stars to use for estimating the PSF model
sigma_ratio = 1.5 # maximum allowable ratio of x/y sigmas (guards against badly asymmetric PSF models)

# source detection
rdnoise = 1.2 # e- Sensitivity mode: 1.2, Dynamic mode: 1.6, Speed mode: 2
gain = 0.25 # e-/ADU Sensitivity mode: 0.25, Dynamic mode: 0.23, Speed mode: 0.85
thresh = 100 # 50 (Sensitivity mode) #10 # threshold (in sigmas) of detection map
emp_detect = True # use an empirical estimate of data noise in place of noise model (overides rdnoise and gain)

# background boxes
nbboxes = 8 # maximum number of background boxes to generate
bbox_size = 128 # single axis size in pixels of square background boxes

real_time_plot = True # toggle whether to plot output in real time
plot_freq = 100 # 1000 (Speed mode) 250 (Sensitivity mode) # [images] how frequent to update the real-time plotting / save the stamp data
pow = 2 # power-law scaling of real time plotted stamps

##### SCENE CHANGE CONTROL - Decisions made at the frequency of plot_freq ############
burn_in = 3 # how many data batches to collect before switching the scene (establishes baseline flux for one of the scene change criteria)
consecutive = 3 # how many consecutive scene change flags before making a switch
scene_change_sky_thresh = 2 # 1.25 (Speed), 2 (Sensitivity) # minimum allowable ratio of source flux / sky flux
scene_change_flux_thresh = 10 # significance of flux change in sigmas
stamp_centrality_thresh = 0.3 # minimum allowable relative central peak flux location during burn-in phase (i.e. 0.5 == centre, 1 == edge, 0.2 == 0.8 == off-centre)
hang_time = 1 # [s] time to wait before reacquiring data after a run has been aborted
######################################################################################

#### AUTOGUIDE ##############
autoguide = False
ag_stamp_size = 128
ag_share_path = 'C:/Users/rgomer/Desktop/share'
ag_dist_thresh = 4
###########################

# NUMBER DEPRECATION WARNINGS?
suppress_numba_warning = True
