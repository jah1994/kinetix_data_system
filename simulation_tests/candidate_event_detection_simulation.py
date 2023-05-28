# imports
from tqdm import tqdm # progressbar
import os
import numpy as np
from astropy.stats import mad_std
#import matplotlib.pyplot as plt
import time

# identify subdirectories containg the results of each run
path_to_data = "E:\McDonald backups"
subdirs = [s for d, s, f in os.walk(path_to_data)][0]
subdirs = [s for s in subdirs if 'results' in s ] # filter out any other directories

subdirs = [s for s in subdirs if 'speed' not in s ] # CHANGE THIS TO TOGGLE BETWEEN SENSITIVY AND SPEED MODE DIRECTORIES
subdirs = [s for s in subdirs if 'rtp_M11' not in s ] # Many false positives in this one

subdirs = [os.path.join(path_to_data, s) for s in subdirs]
print(subdirs)
#subdirs = [s for s in subdirs if 'results_M11' != s ] # Deal with this one separately...
#subdirs = [s for s in subdirs if 'results_M11_Sept20_2' == s ]
#subdirs = [s for s in subdirs if 'results_M11_sensitivity' == s ]

def order_files(subdir):

    # grab all files in the subdir
    files = [f for d, s, f in os.walk(subdir)][0]

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
            elif 'sky' in f and 'batch' + b + '.npy' in f:
                sky_files.append(f)

    return phot_files, seq_files, sky_files

def batch_files(files, arr, path):
    for i,file in enumerate(files):
        arr[i] = np.load(os.path.join(path, file))
    return arr


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


def flag_occultation_candidate(results_list, lc_id, tp_id, data, template, thresh):

    one_pad = np.ones(len(data) - len(template))
    template_padded = np.append(template, one_pad)
    snrt = matched_filter_snr(template_padded, data)

    if np.any(snrt > thresh):
        locations = np.where(snrt > thresh)
        #locations += int(len(template)/2) ## apply shift
        results_list.append([int(lc_id), int(tp_id), snrt[locations], locations[0]])

    #results_list.append([int(lc_id), int(tp_id), np.max(snrt),
    #                    np.where(snrt == np.max(snrt))[0][0]]) #, snrt])


# determine sources > limiting snr
def high_snr_sources(X):

    X_norm = X / np.median(X, axis=0)
    comparison_stars = np.where(np.std(X_norm[:, :], axis=0) < 0.1)[0]

    # if one or less comparison stars fail to meet the above condition, just use the stablest 10 OR brightest 10
    if len(comparison_stars) <= 1:
        #comparison_stars = np.argsort(np.std(X_norm[:, :], axis=0))[:10] # stablest... not always a good choice
        comparison_stars = np.arange(0, 10, 1) # brightest
    wav = np.average(X_norm[:, comparison_stars],
                 #weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
                 weights=1/np.var(X_norm[:, comparison_stars], axis=0), axis=1)

    sources = []
    for i, lc in tqdm(enumerate(X_norm[:, :].T)):

        # detrend data
        lc_ = lc / wav

        # skip lc if noise exceeds some threshold
        if mad_std(lc_) > 0.3:
            continue
        else:
            sources.append(i)

    return sources

############### inject fake occultation signals ##################
from math import hypot, ceil
from math import pi, cos, tan
from scipy.special import jv  # Bessel function
from itertools import product

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
    weights = np.array([0.19570672, 0.25023261, 0.2839096 , 0.21618307, 1. ,
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
    # phi is opposition angle, in degrees
    # vE is Earth's orbital speed, in m/s
    # returns vT, transverse KBO velocity, in m/s

    #return vE * ( 1 - (1./a)**(1/2.))
    phi_ = phi * (np.pi / 180) # degrees -> radians
    return vE * (np.cos(phi_) - ((1./a) * (1 - (np.sin(phi_) ** 2))) ** (1/2))



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


exposure = 0.012 # seconds
startLam = 2e-7 # 4e-7 start of wavelength range
endLam = 11e-7 # end of wavelength range
#objectRad = 500 # object radius, m
#impact = 0 # impact parameter, m
#dist = 100 # object distance, AU
angDi = 0.02 # angular diameter of star, mas
shiftAdj = 0
#curve = genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj)


################################################################

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

temp_bank, temp_params = gen_template_bank(0.1, 1, 5, 505, 0.1, 10)

# cycle over each results subdirectory
x_min = int(input('x_min:'))
x_max = int(input('x_max:'))
for x in range(x_min, x_max):
    for s in subdirs:

        print('Subdirectory:', s)

        if os.path.isfile('analysisSIM/' + str(s.split("\\")[-1]) + 'res' + str(x) + '.npy') is True:
            print('Results file already exists...')
            print('Skipping:', 'analysisSIM/' + str(s.split("\\")[-1]) + 'res' + str(x) + '.npy')
            continue

        # grab files
        phot_files, seq_files, sky_files = order_files(s)

        # check that there's actually data in the subdirectory, and if there isn't, skip
        if len(phot_files) == 0:
            print('No data files found in the subdirectory. Skipping...')
            continue

        # setup dimensions for the data matrix
        fs = np.load(os.path.join(s, phot_files[0])) # use for shape info
        ss = np.load(os.path.join(s, sky_files[0]))
        data = np.zeros((len(phot_files), fs.shape[0], fs.shape[1])) # data matrix
        t = np.zeros((len(seq_files), fs.shape[0])) # observation sequence numbers (i.e. time)
        bkgs = np.zeros((len(sky_files), ss.shape[0], ss.shape[1]))

        # staple together the data in files
        data = batch_files(phot_files, data, s)
        t = batch_files(seq_files, t, s)
        bkgs = batch_files(sky_files, bkgs, s)

        # reshape and change dtype
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).astype(np.float32)
        bkgs = bkgs.reshape(bkgs.shape[0] * bkgs.shape[1], bkgs.shape[2]).astype(np.float32)
        t = t.flatten().astype(np.float32)
        print('Data matrix shape:', data.shape)
        print('Data matrix dytpe:', data.dtype)

        # nsamples * nfeatures
        X = np.copy(data)

        ########## inject fake occultation signals
        #ns = [0, 5, 10, 30, 50, 100]
        #t0s = [10000, 20000, 30000, 40000, 50000, 60000]
        #Nsim = int(X.shape[1] / 2)
        sources = high_snr_sources(X)
        print('Number of sources above SNR thresh:', len(sources))
        print('Sources:', sources)
        radii = [250, 500, 1000, 2500, 5000, 7500, 10000] # m
        dists = [30, 40, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000] # Au
        phis = [0, 10, 50, 75] # degrees
        bs = [0, 100, 500, 1000, 2500, 5000, 7500, 10000] # m

        sim_params = np.empty((len(sources), 6))
        for k in range(len(sources)):
            sim_params[k] = np.array([sources[k], # one event per light curve
                                    np.random.randint(2500, X.shape[0] - 2500, 1),
                                    radii[np.random.randint(0, len(radii), 1)[0]],
                                    dists[np.random.randint(0, len(dists), 1)[0]],
                                    phis[np.random.randint(0, len(phis), 1)[0]],
                                    bs[np.random.randint(0, len(bs), 1)[0]],
            ])

        print(sim_params)
        print(sim_params.shape)
        np.save(os.path.join(os.getcwd(), 'analysisSIM/') + str(s.split("\\")[-1]) + '_sim_params%d.npy' % x, sim_params)

        print('Injecting fake events into sources...')
        for params in sim_params:
            n, t0, radii, dist, phi, b = params
            n, t0 = int(n), int(t0)
            curve = genCurve(exposure, startLam, endLam, radii, b, dist, angDi, shiftAdj, phi)
            t1 = t0 + len(curve)
            X[:, n][t0:t1] += (np.median(X[:, n]) * (curve - 1))



        # systematic variation is a funtion of flux, so normalise out
        X_norm = X / np.median(X, axis=0)
        print(X_norm.shape)


        # compute an average of this systematic variation with the brightest Ncomp sources
        #wav = np.average(X_norm[:, :Ncomp], weights=np.sqrt(np.median(X[:, :Ncomp], axis=0)), axis=1)

        # select all stars with rms < 0.1
        comparison_stars = np.where(np.std(X_norm[:, :], axis=0) < 0.1)[0]

        # if one or less comparison stars fail to meet the above condition, just use the stablest 10 OR brightest 10
        #if len(comparison_stars) <= 1:
        if len(comparison_stars) <= 2:
            #comparison_stars = np.argsort(np.std(X_norm[:, :], axis=0))[:10] # stablest... not always a good choice
            comparison_stars = np.arange(0, 10, 1) # brightest

        print("Number of potential comparison stars:", len(comparison_stars))


        '''
        print("Number of comparison stars:", len(comparison_stars))
        #wav = np.average(X_norm[:, comparison_stars], weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
        wav = np.average(X_norm[:, comparison_stars],
                     #weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
                     weights=np.std(X_norm[:, comparison_stars], axis=0), axis=1)
        '''

        ## generate a template bank of candidate occultation signals
        #temp_bank, temp_params = gen_template_bank(0.2, 0.9, 5, 505, 0.1, 10)
        #temp_bank, temp_params = np.load('temp_bank.npy', allow_pickle=True), np.load('temp_params.npy', allow_pickle=True)
        print("Template bank size:", len(temp_bank))

        ## cycle over each light curve and identifying similarity of each template
        res = [] # stick results in this
        for i, lc in tqdm(enumerate(X_norm[:, :].T)):

            comp_stars = [c for c in comparison_stars if c != i]
            print("Number of comparison stars:", len(comp_stars))
            print(i)
            print(comp_stars)
            #wav = np.average(X_norm[:, comparison_stars], weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
            wav = np.average(X_norm[:, comp_stars],
                         #weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
                         weights=1/np.var(X_norm[:, comp_stars], axis=0), axis=1)

            # detrend data
            lc_ = lc / wav

            # skip lc if noise exceeds some threshold
            if mad_std(lc_) > 0.3:
                continue


            print('Starting matched-filtering...')
            tmf = time.time()
            # roll out the template signals...
            for j, tmplt in enumerate(temp_bank):
                flag_occultation_candidate(res, i, j, lc_, tmplt, thresh=5)
            print('Finished matched-filtering in %.2f seconds.' % (time.time() - tmf))

        # convert results to an array type
        res = np.array(res)
        print('Results array shape:', res.shape)
        np.save(os.path.join(os.getcwd(), 'analysisSIM/') + str(s.split("\\")[-1]) + 'res%d.npy' % x, res)
