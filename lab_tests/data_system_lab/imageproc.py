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

    # minimise loss w.r.t parameters
    res = minimize(loss, params, method='L-BFGS-B')
    sigma_x, sigma_y = res.x[1], res.x[2]
    return sigma_x, sigma_y

def update_r(ref, positions, r, nsigma, lim, out_path):

    # restrict to the lim brightest sources
    nsources = len(positions[:lim])

    # store 2D Gaussian sigmas
    sigma_xs = np.empty(nsources)
    sigma_ys = np.empty(nsources)

    # build a median PSF from the nsources
    for n in range(nsources):

        # source position (x, y)
        pos = positions[n]

        # cutout stamp
        stamp = ref[pos[1] - r : pos[1] + r, pos[0] - r : pos[0] + r]

        sigma_x, sigma_y = fit_gaussian_psf(stamp)
        sigma_xs[n] = sigma_x
        sigma_ys[n] = sigma_y

    # set r as some multiple of nsigma
    sigma_x_, sigma_y_ = np.median(sigma_xs), np.median(sigma_ys)
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
    plt.savefig(os.path.join(out_path, 'psf.png'));

    return rx, ry, sigma_x_, sigma_y_, psf_model


def make_detection_map(ref, psf_model, rdnoise, gain):

    #ref[ref < 0] = 0 # hack!

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
    return dark_stamps, flat_stamps

@jit(nopython=True, parallel=True, nogil=False)
def fluxes_stamps(image, dark_stamps, flat_stamps, positions, nsources, rx, ry):

    #nsources = len(positions) # Numba doesn't work with astropy tables
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



#@jit(nopython=True, parallel=True, nogil=False)
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

        # sum up the flux of the processed pixels in the aperture
        skys[n] = np.median(proc_stamp)

    return skys



def background_boxes(positions, peaks, ref, rx, ry, out_path, q=50, bc=250, N=16):

    # Perform KDE on the positional data, weighted by peak flux
    kde = gaussian_kde(positions.T, weights=peaks, bw_method='scott')

    # Define a grid of points where we want to evaluate the KDE
    print('Evaluating KDE for the scene...')
    t0 = time.perf_counter()
    xmin, xmax, ymin, ymax = min(positions[:,0]), max(positions[:,0]), min(positions[:,1]), max(positions[:,1])
    x_grid, y_grid = np.mgrid[xmin:xmax:3200j, ymin:ymax:3200j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate the KDE on the grid
    kde_values = kde(grid_coords)

    # Reshape the KDE values to match the grid shape
    kde_values = kde_values.reshape(x_grid.shape).T
    print('Finished in %.3f seconds.' % (time.perf_counter() - t0))

    plt.imshow(kde_values, origin='lower')
    plt.savefig(os.path.join(out_path, 'kde.png'));

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
    box_sizes_x = [200, 150, 100]
    box_sizes_y = [200, 150, 100]

    # box centres and radii
    bb_pos = []
    bb_rs = []

    # find candidate regions
    i = 0
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

    return bb_pos, bb_rs