import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
#from scipy.signal import correlate2d
from scipy.signal import fftconvolve
import time


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

def update_r(ref, positions, r, nsigma, lim):

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
    plt.show();

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



def sky_calibration_stamps(dark, flat, bb_pos, bb_rs):
    dark_stamps = [dark[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]] for pos, rs in zip(bb_pos, bb_rs)]
    flat_stamps = [flat[pos[1] - rs[1] : pos[1] + rs[1], pos[0] - rs[0] : pos[0] + rs[0]] for pos, rs in zip(bb_pos, bb_rs)]
    return dark_stamps, flat_stamps


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

        # sum up the flux of the processed pixels in the aperture
        skys[n] = np.median(proc_stamp)

    return skys



def background_boxes(ref):

    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        rx, ry = abs(int((x2 - x1)/2)), abs(int((y2 - y1)/2))
        xc, yc = int((x2 + x1) / 2), int((y2 + y1) / 2)
        bb_pos.append([xc, yc])
        bb_rs.append([rx, ry])

        ## hacky way to keep track of drawn boxes - keep replotting
        ## on a newly generated figure everytime a new box is specified
        ls = 15
        fs = 10

        running_fig = plt.figure(figsize=(20, 20))
        ax = running_fig.add_subplot(111)
        ax.tick_params(axis='both', labelsize=ls)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(ref, origin='lower')
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label("Counts [ADU]", fontsize=ls)
        cbar.ax.tick_params(labelsize=ls)

        for j, (pos, rs) in enumerate(zip(bb_pos, bb_rs)):

            ax.add_patch(patches.Rectangle(xy=(pos[0] - rs[0], pos[1] - rs[1]),
                                           width=2*rs[0], height=2*rs[1], fill=False, label=j, color='red'))
            ax.text(pos[0], pos[1], str(j), fontsize=fs, c='r')
        plt.show();


    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    bb_pos = []
    bb_rs = []

    fig, current_ax = plt.subplots(figsize=(20,20))
    plt.imshow(ref, origin='lower')

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    bb_pos, bb_rs = np.array(bb_pos), np.array(bb_rs)

    return bb_pos, bb_rs


@jit(nopython=True, parallel=True, nogil=False)
def fluxes_and_uncertainties_stamps(image, dark_stamps, flat_stamps, positions, nsources, rx, ry, rdnoise, gain):

    #nsources = len(positions) # Numba doesn't work with astropy tables
    fluxes = np.zeros(nsources)
    fluxes_vars = np.zeros(nsources)

    for n in prange(nsources):

        # source position (x, y)
        pos = positions[n]

        # cutout the stamp around the source position and reduce pixels
        #stamp = image[pos[0] - r : pos[0] + r, pos[1] - r : pos[1] + r]
        stamp = image[pos[1] - ry : pos[1] + ry, pos[0] - rx : pos[0] + rx]

        # dark subtract and flat correct the stamp
        proc_stamp = (stamp - dark_stamps[n]) / flat_stamps[n]

        # sum up the flux of the processed pixels in the aperture
        fluxes[n] = np.sum(proc_stamp)

        # compute pixel variances
        var = (sigma0 / flat_stamps[n])**2 + proc_stamp / (gain * flat_stamps[n])

        # sum variances for total on measured flux
        fluxes_vars[n] = np.sum(var)

    return fluxes, fluxes_vars
