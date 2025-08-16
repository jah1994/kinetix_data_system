import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
#from scipy.signal import correlate2d
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from scipy.ndimage import zoom
import time
import os

@jit(nopython=True, parallel=True, nogil=False)
def proc(data, dark, flat):
    data_proc = (data - dark) / flat
    return data_proc

#@jit(nopython=True, parallel=True, nogil=False)
# todo: we-rite this so it's numba compliant. numba doesn't support axis argument in np.mean
def time_average(array):
    tav = np.mean(array, axis=0)
    return tav

def fit_gaussian_psf(data):

    # generate psf model stamp
    shape = data.shape
    nx, ny = shape

    # line associated with scene change criterion, as square stamps required
    if nx != ny:
        max_length = max([nx, ny])
        nx, ny = max_length, max_length
    assert nx == ny

    # define the x/y grid
    xg, yg = np.meshgrid(range(nx), range(ny))

    # loss function
    def loss(params):
        A, sigma_x, sigma_y, b, xc, yc = params
        prediction = np.exp(-0.5 * (((xg - xc) / sigma_x) ** 2
                                 +  ((yg - yc) / sigma_y) ** 2)
                           )

        prediction *= A
        prediction += b
        return np.sum((prediction - data)**2)

    # initialise parameters
    params = np.array([10., 5., 5., 0., (nx - 1)/2, (ny - 1)/2])

    # bounds for parameters
    #bounds = [(1e-3, 1e3), (1, 30), (1, 30), (-1e3, 1e3), (1, nx), (1, ny)]
    bounds = [(0, 1e9), (1, 30), (1, 30), (-1e9, 1e9), (1, nx), (1, ny)]

    # minimise loss w.r.t parameters
    res = minimize(loss, params, method='L-BFGS-B', bounds=bounds)
    print('PSF model parameter estimates:', res.x)
    sigma_x, sigma_y, xc, yc = res.x[1], res.x[2], res.x[4], res.x[5]
    return sigma_x, sigma_y, xc, yc

def update_r(ref, positions, r, nsigma, lim, img_path):

    # restrict to the lim brightest sources
    nsources = len(positions[:lim])

    # store 2D Gaussian sigmas
    sigma_xs = np.empty(nsources)
    sigma_ys = np.empty(nsources)
    xcs = np.empty(nsources)
    ycs = np.empty(nsources)

    # build a median PSF from the nsources
    for n in range(nsources):

        # source position (x, y)
        pos = positions[n]

        # cutout stamp
        stamp = ref[pos[1] - r : pos[1] + r, pos[0] - r : pos[0] + r]

        # fit model, returning standard deviation and centres
        sigma_x, sigma_y, xc, yc = fit_gaussian_psf(stamp)
        sigma_xs[n] = sigma_x
        sigma_ys[n] = sigma_y
        xcs[n] = xc
        ycs[n] = yc

    # set r as some multiple of nsigma
    sigma_x_, sigma_y_ = np.median(sigma_xs), np.median(sigma_ys)
    xcs_, ycs_ = np.median(xcs), np.median(ycs)
    rx, ry = int(nsigma * sigma_x_), int(nsigma * sigma_y_)

    # generate psf model stamp
    nx, ny = int(5 * sigma_x_), int(5 * sigma_y_)
    xg, yg = np.meshgrid(range(nx), range(ny))
    xc, yc = (int((nx - 1)/2), int((ny - 1)/2))
    psf_model = np.exp(-0.5 * (((xg - xc) / sigma_x_) ** 2
                              + ((yg - yc) / sigma_y_) ** 2)
                        )
    psf_model /= np.sum(psf_model)

    plt.imshow(psf_model)
    plt.title('PSF Model')
    plt.savefig(os.path.join(img_path, 'psf.png'))
    plt.close();

    return rx, ry, sigma_x_, sigma_y_, xcs_, ycs_, psf_model


def make_detection_map(ref, psf_model, rdnoise, gain):

    # match-filter reference with the PSF Model to generate a detection map
    print('Cross-correlating reference image with the PSF model... this could take a while.')
    #M = correlate2d(ref, psf_model, mode='same') # TODO: multi-core version of this??
    M = fftconvolve(ref, psf_model, mode='same') # TODO: multi-core version of this??

    phi = np.sqrt(np.sum(psf_model ** 2)) # normalisation constant
    D = M / (phi ** 2) # detection map

    # approximate pixel uncertainties
    var = (rdnoise / gain) ** 2 + (ref / gain)
    sigma = np.sqrt(var)

    # per-pixel error in detection map
    sigma_D = sigma / phi

    return D / sigma_D

def make_detection_map_empirical(ref, psf_model, mad_std):

    # match-filter reference with the PSF Model to generate a detection map
    print('Cross-correlating reference image with the PSF model... this could take a while.')
    M = fftconvolve(ref, psf_model, mode='same') # TODO: multi-core version of this??

    phi = np.sqrt(np.sum(psf_model ** 2)) # normalisation constant
    D = M / (phi ** 2) # detection map

    # per-pixel error in detection map
    sigma_D = mad_std / phi

    return D / sigma_D


def calibration_stamps(dark, flat, positions, rx, ry):
    dark_stamps = [dark[pos[1] - ry : pos[1] + ry, pos[0] - rx : pos[0] + rx] for pos in positions]
    flat_stamps = [flat[pos[1] - ry : pos[1] + ry, pos[0] - rx : pos[0] + rx] for pos in positions]
    return np.array(dark_stamps), np.array(flat_stamps)


def sky_calibration_stamps(dark, flat, bb_pos, bb_rs):
    dark_stamps = [dark[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]] for pos, rs in zip(bb_pos, bb_rs)]
    flat_stamps = [flat[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]] for pos, rs in zip(bb_pos, bb_rs)]
    return np.array(dark_stamps), np.array(flat_stamps)

@jit(nopython=True, parallel=True, nogil=False)
def fluxes_stamps(image, dark_stamps, flat_stamps, positions, nsources, rx, ry):
    fluxes = np.zeros(nsources)
    for n in prange(nsources):
        # source position (x, y)
        pos = positions[n]
        # cutout the stamp around the source position and reduce pixels
        stamp = image[pos[1] - ry : pos[1] + ry, pos[0] - rx : pos[0] + rx]
        # dark subtract and flat correct the stamp
        proc_stamp = (stamp - dark_stamps[n]) / flat_stamps[n]
        # sum up the flux of the processed pixels in the aperture
        fluxes[n] = np.sum(proc_stamp)
    return fluxes

@jit(nopython=True, parallel=True, nogil=False)
def sky_stamps(image, dark_stamps, flat_stamps, nboxes, bb_pos, bb_rs):
    skys = np.zeros(nboxes)
    for n in prange(nboxes):
        # box position (x, y)
        pos = bb_pos[n]
        # box radii (x, y)
        rs = bb_rs[n]
        # cutout the stamp around the source position and reduce pixels
        stamp = image[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]]
        # dark subtract and flat correct the stamp
        proc_stamp = (stamp - dark_stamps[n]) / flat_stamps[n]
        # compute the median of the flux of the processed pixels in the sky stamp
        skys[n] = np.median(proc_stamp)
    return skys

@jit(nopython=True, parallel=True, nogil=False)
def fluxes_stamps_nocal(image, positions, nsources, rx, ry):
    fluxes = np.zeros(nsources)
    for n in prange(nsources):
        # source position (x, y)
        pos = positions[n]
        # cutout the stamp around the source position and reduce pixels
        stamp = image[pos[1] - ry : pos[1] + ry, pos[0] - rx : pos[0] + rx]
        # sum up the flux of the processed pixels in the aperture
        fluxes[n] = np.sum(stamp)
    return fluxes

@jit(nopython=True, parallel=True, nogil=False)
def sky_stamps_nocal(image, nboxes, bb_pos, bb_rs):
    skys = np.zeros(nboxes)
    for n in prange(nboxes):
        # box position (x, y)
        pos = bb_pos[n]
        # box radii (x, y)
        rs = bb_rs[n]
        # cutout the stamp around the source position and reduce pixels
        stamp = image[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]]
        # compute the median of the flux of the processed pixels in the sky stamp
        skys[n] = np.median(stamp)
    return skys

def background_boxes(positions, peaks, ref, rx, ry, img_path, bbox_size, q=50, bc=250, N=16):

    # Perform KDE on the positional data, weighted by peak flux
    kde = gaussian_kde(positions.T, weights=peaks, bw_method='scott')

    # Define a grid of points where we want to evaluate the KDE
    print('Evaluating KDE for the scene...')
    t0 = time.perf_counter()
    xmin, xmax, ymin, ymax = min(positions[:,0]), max(positions[:,0]), min(positions[:,1]), max(positions[:,1])
    x_grid, y_grid = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate the KDE on the grid
    kde_values = kde(grid_coords)

    # Reshape the KDE values to match the grid shape
    kde_values = kde_values.reshape(x_grid.shape).T

    # Define the upsampling factor (e.g., 2x or 3x)
    upsample_factor = 8

    # Perform the upsampling using scipy.ndimage.zoom
    kde_values = zoom(kde_values, (upsample_factor, upsample_factor))
    print('Finished in %.3f seconds.' % (time.perf_counter() - t0))

    plt.imshow(kde_values, origin='lower')
    plt.savefig(os.path.join(img_path, 'kde.png'))
    plt.close();

    # q kde percentile to restrict search space
    kde_q = np.percentile(kde_values.flatten(), q=q)
    kde_values[np.where(kde_values < kde_q)] = np.nan

    # delta function source postions map
    #df_map = np.zeros(kde_values.shape)
    #df_map[positions[:,0], positions[:,1]] = 1

    # map apertures to be avoided by background regions
    df_map = np.zeros(ref.shape)
    for pos_x, pos_y in positions:
        df_map[pos_x - rx : pos_x + rx, pos_y - ry : pos_y + ry] = 1


    # quasi-random search over space
    #box_sizes_x = [200, 150, 100]
    #box_sizes_y = [200, 150, 100]
    box_sizes_x = [bbox_size]
    box_sizes_y = [bbox_size]

    # box centres and radii
    bb_pos = []
    bb_rs = []

    # find candidate regions (if unable to find N, find as many as possible in 5 seconds)
    i = 0
    t = time.time()
    while i < N:

        # find coordinates within the search region
        xc, yc = np.random.randint(bc, ref.shape[0] - bc), np.random.randint(bc, ref.shape[1] - bc)
        if np.isnan(kde_values[xc, yc]) == True:

            # are any sources within this region, defined by the box_size
            for box_size_x in box_sizes_x:
                for box_size_y in box_sizes_y:
                    candidate_region = df_map[xc - box_size_x:xc + box_size_x, yc -  box_size_y:yc + box_size_y]

                    if np.any(candidate_region) == 1:
                        continue
                    else:
                        bb_pos.append([xc, yc])
                        bb_rs.append([box_size_x, box_size_y])

                        # update the df image with new background regions
                        df_map[xc - box_size_x:xc + box_size_x, yc -  box_size_y:yc + box_size_y] = 1
                        i += 1
        if time.time() - t >= 5:
            break

    return np.array(bb_pos), np.array(bb_rs)

def compute_mode(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    mode_value = unique_elements[max_count_index]
    mode_count = counts[max_count_index]
    return mode_value, mode_count

# add two numpy arrays
@jit(nopython=True, parallel=True, nogil=False)
def add_arrays(arr1, arr2):
    return arr1 + arr2
