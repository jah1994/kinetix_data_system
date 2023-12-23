# imports
import os
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt

# load data and generate a matrix of observations
path = "E:/McDonald backups/results_M11_Sept20_2"
files = [f for d, s, f in os.walk(path)][0]

# order the files
phot_files = []
seq_files = []
sky_files = []
batch_ids = np.arange(0, 1000).astype(str)
for b in batch_ids:
    for f in files:
        if 'photometry' in f and 'batch' + b + '.npy' in f:
            phot_files.append(f)
        elif 'seq' in f and 'batch' + b + '.npy' in f:
            seq_files.append(f)

n = 0 # use as reference for array dimensions
fs = np.load(os.path.join(path, phot_files[n])) # use for shape info
data = np.zeros((len(phot_files), fs.shape[0], fs.shape[1])) # data matrix
t = np.zeros((len(seq_files), fs.shape[0])) # observation sequence numbers (i.e. time)

# staple together the data in files
def batch_files(files, arr):
    for i,file in enumerate(files):
        arr[i] = np.load(os.path.join(path, file))
    return arr

data = batch_files(phot_files, data)
t = batch_files(seq_files, t)

# reshape and change dtype
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).astype(np.float32)
t = t.flatten().astype(np.float32)


### simulated occultation signal parameters
from math import hypot, ceil
from math import pi, cos, tan
from scipy.special import jv  # Bessel function
from itertools import product
import time

class memoize:

    """ Memoization decorator to cache repeatedly-used function calls """

    # stock code from http://avinashv.net

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


@memoize
def lommel(n, a, b):

    """ Calculates the nth lommel function """

    U = 0
    for k in range(0, 100000):
        sum = ((-1)**k * (a/b)**(n+2*k) * jv(n+2*k, pi*a*b))
        U += sum
        if abs(sum) < 0.00001:
            return U
    raise ValueError("Failure to converge")


@memoize
def generatePoints(starR):

    """ Models star as an array of uniformly distributed point sources """

    if starR == 0:  # model as point source
        return np.array([(0,0)])
    n = 5  # number of points to model 1D radius of star
    pairs = np.array([item for item in product(np.linspace(-starR, starR, 2*n-1), repeat=2) if hypot(item[0], item[1]) <= starR])
    return pairs


def diffractionCalc(r, p, starR, lam, D, b):

    """ Analytically calculates intensity at a given distance from star centre """

    # r is distance between line of sight and centre of the disk, in fresnel scale units
    # p is radius of KBO, in fresnel scale units
    pts = generatePoints(starR)
    r = fresnel(r, lam, D)
    res = 0
    effR = np.round(np.hypot((r - pts[:, 0]), (pts[:, 1]-b)), 2)
    coslist = np.cos(0.5*pi*(effR**2 + p**2))
    sinlist = np.sin(0.5*pi*(effR**2 + p**2))
    l = len(pts)
    for n in range(0, l):
        if effR[n] > p:
            U1 = lommel(1, p, effR[n])
            U2 = lommel(2, p, effR[n])
            res += (1 + (U2 ** 2) + (U1 ** 2) + 2*(U2*coslist[n] - U1*sinlist[n]))
        elif effR[n] == p:
            res += (0.25 * ((jv(0, pi*p*p)**2) + 2*cos(pi*p*p)*jv(0, pi*p*p) + 1))
        else:
            res += ((lommel(0, effR[n], p)**2) + (lommel(1, effR[n], p) ** 2))
    return res / l


def fresnel(x, lam, D):

    """ Converts value to fresnel scale units """
    return x / (lam*D/2.)**(1/2.)


def generateKernel(lam, objectR, b, D, starR):

    """ Calculates the light curve at a given wavelength """

    p = fresnel(objectR, lam, D)  # converting KBO radius to Fresnel Scale
    s = fresnel(starR, lam, D)  # converting effective star radius to Fresnel Scale
    b = fresnel(b, lam, D)
    #r = 25000.  # distance between line of sight and centre of the disk in m
    r = 50000
    z = [diffractionCalc(j, p, s, lam, D, b) for j in np.arange(-r, r, 10)]
    return z


def defineParam(startLam, endLam, objectR, b, D, angDi):

    """ Simulates light curve for given parameters """

    # startLam: start of wavelength range, m
    # endLam: end of wavelength range, m
    # objectR: radius of KBO, m
    # b: impact parameter, m
    # D: distance from KBO to observer, in AU
    # angDi: angular diameter of star, mas
    # Y: light profile during diffraction event

    D *= 1.496e+11 # converting to metres
    starR = effStarRad(angDi, D)
    n = 18
    weights = np.array([0.19570672, 0.25023261, 0.2839096 , 0.21618307, 1.        ,
       0.99264115, 0.7642776 , 0.60915081, 0.47424386, 0.36451124,
       0.27732518, 0.20631212, 0.14537288, 0.09838498, 0.06365699,
       0.03863518, 0.02244682, 0.00967328])
    if endLam == startLam:
        Y = generateKernel(startLam, objectR, b, D, starR)
    else:
        step = (endLam-startLam) / n
        Y = np.array([generateKernel(lam, objectR, b, D, starR) for lam in np.arange(startLam, endLam, step)])
        #Y = np.sum(Y, axis=0)
        #Y /= n
        Y = np.average(Y, weights=weights, axis=0)
    return Y


def effStarRad(angDi, D):

    """ Determines projected star radius at KBO distance """

    angDi /= 206265000.  # convert to radians
    return D * tan(angDi / 2.)


def vT(a, phi, vE):

    """ Calculates transverse velocity of KBO """

    # a is distance to KBO, in AU
    # phi is opposition angle, degrees
    # vE is Earth's orbital speed, in m/s
    # returns vT, transverse KBO velocity, in m/s

    phi_ = phi * (np.pi / 180) # degrees -> radians
    #return vE * ( 1 - (1./a)**(1/2.))
    return vE * (np.cos(phi_) - ((1./a) * (1 - (np.sin(phi_) ** 2)))**(1/2))


def integrateCurve(exposure, curve, totTime, shiftAdj):

    """ Reduces resolution of simulated light curve to match what would be observed for a given exposure time """

    curve = np.array(curve)
    timePerFrame = totTime / len(curve)
    numFrames = roundOdd(exposure/timePerFrame)
    if shiftAdj < 0:
        shiftAdj += 1
    shift = ((len(curve) / 2)% numFrames) - (numFrames-1)/2
    while shift < 0:
        shift += numFrames
    shift += int(numFrames*shiftAdj)
    for index in np.arange((numFrames-1)/2 + shift, len(curve)-(numFrames-1)/2, numFrames):
        indices = range(int(index - (numFrames-1)/2), int(index+1+(numFrames-1)/2))
        av = np.average(curve[indices])
        curve[indices] = av
    last = indices[-1]+1  # bins leftover if light curve length isn't divisible by exposure time
    shift = int(shift)
    curve[last:] = np.average(curve[last:])
    curve[:shift] = np.average(curve[:shift])
    return curve, numFrames


def roundOdd(x):

    """ Rounds x to the nearest odd integer """

    x = ceil(x)
    if x % 2 == 0:
        return int(x-1)
    return int(x)


def genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj, phi):

    """ Convert diffraction pattern to time series """

    velT = vT(dist, phi, 29800)
    curve = defineParam(startLam, endLam, objectRad, impact, dist, angDi)
    n = len(curve)*10./velT
    curve, num = integrateCurve(exposure, curve, n, shiftAdj)
    return curve[::num]
    #return curve

exposure = 0.012 # seconds
startLam = 2e-7 # 4e-7 start of wavelength range
endLam = 11e-7 # end of wavelength range
#objectRad = 500 # object radius, m
impact = 0 # impact parameter, m
#dist = 100 # object distance, AU
angDi = 0.02 # angular diameter of star, mas
shiftAdj = 0
phi = 0

objectRads = [250, 500, 1250, 2500]
dists = [40, 40, 1000, 1000]
#curve = genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj)

def gen_template_bank(dmin, dmax, tmin, tmax, del_d, del_t):

    nd = int((dmax - dmin) / del_d)
    nt = int((tmax - tmin) / del_t)

    ds = np.linspace(dmin, dmax, nd)
    ts = np.linspace(tmin, tmax, nt).astype(int)

    # add in some finer delta_ts at small timescales
    ts = np.append(ts, np.arange(2, 21, 1)).astype(int)

    bank = [] # list as templates have different lengths
    params = [] # t and d for each template
    for d in ds:
        for t in ts:
            template = np.ones(t + 10) # buffer of 10 - U-shaped dip
            template[5 : t + 5] =  1 - d
            bank.append(template)
            params.append([d, t])

    return bank, params

def matched_filter_snr(template, data):

    # fourier transform template
    tf = np.fft.fft(template)

    # fourier transform data
    df = np.fft.fft(data)

    # compute (unnormalised) matched filter in frequency domain
    mff = tf.conjugate() * df

    # inverse fourier transform to time domain
    mft = np.fft.ifft(mff)
    mft = np.abs(mft)

    # matched filter (empirical) SNR in time domain
    snrt = (mft - np.median(mft)) / mad_std(mft)

    return snrt

# U-shaped template bank
temp_bank, temp_params = gen_template_bank(0.1, 1, 5, 505, 0.1, 10)
print('Number of templates:', len(temp_bank))

ymins = [-4, -10, -3, -3]
ymaxs = [10, 25, 6, 18]

for objectRad, dist, ymin, ymax in zip(objectRads, dists, ymins, ymaxs):
    print('Object radius [m]:', objectRad)
    print('Object distance [au]:', dist)

    # KBO == Blue, OCO == Orange
    if dist > 60:
        buffer = 1000
        color = "tab:orange"
    else:
        buffer = 500
        color = "tab:blue"

    # generate observatons matrix (n_samples x n_features)
    X = np.copy(data)

    # inject the fake occultation signal
    curve = genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj, phi)
    #print(len(curve))

    n = 31
    t0 = 100000
    t1 = t0 + len(curve)
    X[:, n][t0:t1] += (np.median(X[:, n]) * (curve - 1))

    # variation is a funtion of flux, so normalise out
    X_norm = X / np.median(X, axis=0)

    comparison_stars = np.where(np.std(X_norm[:, :], axis=0) < 0.1)[0]
    comp_stars = [c for c in comparison_stars if c != n]
    print("Number of comparison stars:", len(comp_stars))
    #wav = np.average(X_norm[:, comparison_stars], weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
    wav = np.average(X_norm[:, comp_stars],
                 #weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
                 weights=1/np.var(X_norm[:, comp_stars], axis=0), axis=1)

    def flag_occultation_candidate(results_list, lc_id, tp_id, time_series, template):

        one_pad = np.ones(len(time_series) - len(template))
        template_padded = np.append(template, one_pad)
        snrt = matched_filter_snr(template_padded, time_series)

        results_list.append([int(lc_id), int(tp_id), np.max(snrt),
                             np.where(snrt == np.max(snrt))[0][0], snrt])

    res = [] # stick results in this

    # detrend data
    lc = X_norm[:, n].T
    lc_ = lc /  wav

    i = 0
    for j, tmplt in enumerate(temp_bank):
        flag_occultation_candidate(res, i, j, lc_[t0 - buffer : t1 + buffer] , tmplt)

    res = np.array(res, dtype=object)

    # plot up the results
    fontsize = 72
    lw = 3
    pad = 25
    #snrt_max = res[:,4][np.where(res[:,2] == np.max(res[:,2]))[0][0]]
    template_id = res[:,1][np.where(res[:,2] == np.max(res[:,2]))[0][0]]
    template_params = temp_params[template_id]

    # best matching template in the known event window
    template = temp_bank[template_id]
    one_pad = np.ones(len(lc_) - len(template))
    template_padded = np.append(template, one_pad)
    snrt_max = matched_filter_snr(template_padded, lc_)

    # MF with the true occultation signal
    curve_padded = np.append(curve, np.ones(len(lc_) - len(curve)))
    snrt_curve = matched_filter_snr(curve_padded, lc_)


    nrows, ncols = 2, 2
    fig, ax = plt.subplots(figsize=(72, 36), nrows=nrows, ncols=2)
    for i in range(nrows):

        if i == 0:

            ax[i][0].plot(t[t0 - buffer : t1 + buffer],
                       X_norm[:, n].T[t0 - buffer : t1 + buffer] / wav[t0 - buffer : t1 + buffer], c='k', lw=lw, alpha=0.5)
            ax[i][0].plot(t[t0 - buffer:t1 + buffer], np.pad(curve, buffer, constant_values=1), c=color, lw=lw)
            ax[i][0].set_ylabel('$I_{\mathrm{Corr}}(t)$', fontsize=fontsize)
            ax[i][0].axes.xaxis.set_ticklabels([])
            ax[i][0].tick_params(labelsize=fontsize-10, pad=pad)
            ax[i][0].set_ylim(0, 1.4)
            ax[i][0].set_xlim(t[t0 - buffer], t[t1 + buffer])
            ax[i][0].grid()

            ax[i][1].plot(t[t0 - buffer : t1 + buffer],
                       X_norm[:, n].T[t0 - buffer : t1 + buffer] / wav[t0 - buffer : t1 + buffer], c='k', lw=lw, alpha=0.5)
            ax[i][1].plot(t[t0 - buffer:t1 + buffer], np.pad(curve, buffer, constant_values=1), c=color, lw=lw)
            ax[i][1].set_ylabel('$I_{\mathrm{Corr}}(t)$', fontsize=fontsize)
            ax[i][1].axes.xaxis.set_ticklabels([])
            ax[i][1].tick_params(labelsize=fontsize-10, pad=pad)
            ax[i][1].set_ylim(0, 1.4)
            ax[i][1].set_xlim(t[t0 - buffer], t[t1 + buffer])
            ax[i][1].grid()

        elif i == 1:

            ax[i][0].plot(t[t0 - int(buffer + len(curve)/2) : t1 + buffer] + int(len(curve)) / 2,
                        snrt_curve[t0 - int(buffer + len(curve)/2) : t1 + buffer], c='tab:green', lw=lw)
            ax[i][0].set_ylabel('$\sigma(t)$', fontsize=fontsize)
            #if ymin == -10:
            #    ax[i][0].tick_params(labelsize=fontsize-10, pad=pad)
            #else:
            #    ax[i][0].tick_params(labelsize=fontsize-10)
            ax[i][0].tick_params(labelsize=fontsize-10, pad=pad)
            ax[i][0].set_xlim(t[t0 - buffer], t[t1 + buffer])
            ax[i][0].set_ylim(ymin, ymax)
            ax[i][0].grid()

            ax[i][0].set_xlabel('$t$ [Exposure number]', fontsize=fontsize)

            ax[i][1].plot(t[t0 - int(buffer + len(curve)/2) : t1 + buffer] + template_params[1],
                        snrt_max[t0 - int(buffer + len(curve)/2) : t1 + buffer], c='tab:green', lw=lw)
            ax[i][1].set_ylabel('$\sigma(t)$', fontsize=fontsize)
            #if ymin == -10:
            #    ax[i][1].tick_params(labelsize=fontsize-10, pad=pad)
            #else:
            #    ax[i][1].tick_params(labelsize=fontsize-10)
            ax[i][1].tick_params(labelsize=fontsize-10, pad=pad)
            ax[i][1].set_xlim(t[t0 - buffer], t[t1 + buffer])
            ax[i][1].set_ylim(ymin, ymax)
            ax[i][1].grid()

            ax[i][1].set_xlabel('$t$ [Exposure number]', fontsize=fontsize)


        # Add time in seconds to x-axis
        ax2 = ax[0][i].twiny()
        ax2.set_xlim(ax[0][i].get_xlim())  # Set the same

        # Customize the additional x-axis (change these according to your requirements)
        if dist > 60:
            t_grid = np.array([99500, 1e5, 100500, 101001, 101500])
            ax2.set_xticks(t_grid)
            ax2.set_xticklabels((12e-3 * t_grid).astype(np.int64))
        else:
            t_grid = np.array([1e5, 100200, 100400, 100600, 100800, 101000, 101200])
            ax2.set_xticks(t_grid)
            ax2.set_xticklabels(np.round((12e-3 * t_grid), 1))

        ax2.set_xlabel('Time [seconds]', fontsize=fontsize, labelpad=15)  # Label for the additional x-axis
        ax2.tick_params(labelsize=fontsize-10, pad=pad)  # Adjust tick label size if needed


    plt.savefig('figures/MF_%d_%d_closeup_v2.png' % (objectRad, dist), bbox_inches='tight')
