# imports
import os
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt

# staple together the data in files
def batch_files(files, arr):
    for i,file in enumerate(files):
        arr[i] = np.load(os.path.join(dir, file))
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

def gen_template_bank(dmin, dmax, tmin, tmax, del_d, del_t):

    nd = int((dmax - dmin) / del_d)
    nt = int((tmax - tmin) / del_t)

    ds = np.linspace(dmin, dmax, nd)
    ts = np.linspace(tmin, tmax, nt).astype(int)

    bank = [] # list as templates have different lengths
    params = [] # t and d for each template
    for d in ds:
        for t in ts:
            template = np.ones(t + 10) # buffer of 10 - U-shaped dip
            template[5 : t + 5] =  1 - d
            bank.append(template)
            params.append([d, t])

    return bank, params

#temp_bank, temp_params = gen_template_bank(0.1, 1, 5, 505, 0.1, 10)


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
    n = 20
    if endLam == startLam:
        Y = generateKernel(startLam, objectR, b, D, starR)
    else:
        step = (endLam-startLam) / n
        Y = np.array([generateKernel(lam, objectR, b, D, starR) for lam in np.arange(startLam, endLam, step)])
        Y = np.sum(Y, axis=0)
        Y /= n
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


# Fixed parameters
exposure = 0.012 # seconds
startLam = 6e-7 # 4e-7 start of wavelength range
endLam = 6e-7 # end of wavelength range
angDi = 0.02 # angular diameter of star, mas
shiftAdj = 0


path_to_data = "E:\McDonald backups"
dirs = [s for d, s, f in os.walk(path_to_data)][0]
#dirs = [s for d, s, f in os.walk(os.getcwd())][0]
dirs = [s for s in dirs if 'results' in s ] # filter out any other directories
dirs = [s for s in dirs if 'speed' not in s ] # CHANGE THIS TO TOGGLE BETWEEN SENSITIVY AND SPEED MODE DIRECTORIES
dirs = [s for s in dirs if 'rtp_M11' not in s ]
dirs = [os.path.join(path_to_data, s) for s in dirs]
print(dirs)

files = [f for d, s, f in os.walk(os.path.join(os.getcwd(), 'analysisSIM'))][0]

files_res = [f for f in files if 'res' in f ]
files_res = [f for f in files_res if 'sim_params' not in f ]
files_res = [f for f in files_res if 'speed' not in f ] # CHANGE THIS TO TOGGLE BETWEEN SENSITIVY AND SPEED MODE DIRECTORIES
files_sim_params = [f for f in files if 'sim_params' in f ]
files_sim_params = [f for f in files_sim_params if 'speed' not in f ]

# file count iterator
#xs = [0, 1, 10, 11]
xmin = int(input('xmin:'))
xmax = int(input('xmax:'))

for x in range(xmin, xmax):

    # cycle over result directories for each run
    for dir in dirs:

        # if results file already analysed
        #if os.path.isfile('analysisSIM/results/' +  dir + '_recovered_' + str(x) + '_noFPreject.npy') is True:
        if os.path.isfile('analysisSIM/results/' +  dir.split('\\')[-1] + '_recovered_' + str(x) + '.npy') is True:
            continue

        # find the associated matched filter results and simulation pararmeter information
        '''
        for f_res in files_res:
            if f_res == dir + 'res' + str(x) + '.npy':
                break
        for f_param in files_sim_params:
            if f_param == dir + '_sim_params' + str(x) + '.npy':
                break
        '''
        try:
            f_res = files_res[np.where(np.array(files_res) == dir.split('\\')[-1] + 'res' + str(x) + '.npy')[0][0]]
            f_param = files_sim_params[np.where(np.array(files_sim_params) == dir.split('\\')[-1] + '_sim_params' + str(x) + '.npy')[0][0]]
        except IndexError:
            continue
        # make sure the matched filter file corresponds to the correct paramter file
        if (f_res.split('res' + str(x))[0] == f_param.split('_sim_params' + str(x))[0]) == False:
            continue

        print('Files:', f_res, f_param)

        res = np.load(os.path.join('analysisSIM', f_res), allow_pickle=True)
        sim_params = np.load(os.path.join('analysisSIM', f_param), allow_pickle=True)

        print('Shapes:', res.shape, sim_params.shape)

        # load data and generate a matrix of observations
        lc_files = [f for d, s, f in os.walk(dir)][0]

        # order the files
        phot_files = []
        seq_files = []
        sky_files = []
        batch_ids = np.arange(0, 1000).astype(str)
        for b in batch_ids:
            for f in lc_files:
                if 'photometry' in f and 'batch' + b + '.npy' in f:
                    phot_files.append(f)
                elif 'seq' in f and 'batch' + b + '.npy' in f:
                    seq_files.append(f)
                elif 'sky' in f and 'batch' + b + '.npy' in f:
                    sky_files.append(f)

        n = 0 # use as reference for array dimensions
        fs = np.load(os.path.join(dir, phot_files[n])) # use for shape info
        ss = np.load(os.path.join(dir, sky_files[n]))
        data = np.zeros((len(phot_files), fs.shape[0], fs.shape[1])) # data matrix
        t = np.zeros((len(seq_files), fs.shape[0])) # observation sequence numbers (i.e. time)
        bkgs = np.zeros((len(sky_files), ss.shape[0], ss.shape[1]))


        data = batch_files(phot_files, data)
        t = batch_files(seq_files, t)
        bkgs = batch_files(sky_files, bkgs)

        # reshape and change dtype
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).astype(np.float32)
        bkgs = bkgs.reshape(bkgs.shape[0] * bkgs.shape[1], bkgs.shape[2]).astype(np.float32)
        t = t.flatten().astype(np.float32)

        X = np.copy(data)

        # sources with candidate event detections
        source_ids = np.unique(res[:,0])

        output = []
        for i in source_ids:

            try:
                res_ = np.array(res)[np.array(res)[:,0] == i]
            except IndexError:
                continue

            if len(res_) == 0:
                continue

            # identify peak SNR in a given window
            locations = res_[:,3]
            snr_peaks = res_[:,2]
            template_ids = res_[:,1]

            # unravel on locations and snr peaks from each and every template
            locs = []
            snrs = []
            for loc, snr_peak in zip(locations, snr_peaks):
                for l, sp in zip(loc, snr_peak):
                    locs.append(l) # all locations of all snr peaks from all templates
                    snrs.append(sp) # all snr peaks from all templates
            locs = np.array(locs)
            unique_locs = np.unique(locs) # to cut down iteration time below...
            snrs = np.array(snrs)

            window = 1000 # search region is loc +- window
            snr_maxs = []
            loc_check = []
            t_ids = []

            # divide time series into window sized grid in which to search for peaks
            grid = np.arange(0, X.shape[0], window) + (window/2)
            grid = grid.astype(int)

            #for uloc in unique_loc:
            for uloc in grid:

                # Already searched for peaks in this region?
                #if (np.abs(np.array(loc_check) - uloc) < window).any() == True:
                #    continue

                #snr_max = np.max(snrs[np.where(np.abs(locs - uloc) < window)])
                try:
                    snr_max = np.max(snrs[np.where(np.abs(locs - uloc) < (window/2))])
                except ValueError: # zero-sized array if no peaks found in window
                    continue
                loc_max = locs[np.where(snrs == snr_max)]

                # Already seen this candidate occultation?
                if snr_max in snr_maxs:
                    continue

                # Threshold?
                if snr_max < 5:
                    continue

                #print(snr_max, loc_max)

                # find the template associated with this particular snr peak
                for sp,snrp in enumerate(snr_peaks):
                    if snr_max in snrp:
                        template_id = template_ids[sp]


                # record dip info to prevent doubling up
                snr_maxs.append(snr_max)
                loc_check.append(loc_max[0])
                t_ids.append(template_id)

            output.append((i, t_ids, loc_check, snr_maxs))



        output = np.array(output, dtype=object)
        print('Output shape:' , output.shape)


        # find isolated loc_max values i.e. common to just a single source
        # unravel all loc_max
        loc_maxs = []
        snr_maxs_ = []
        for iterator1, locs in enumerate(output[:,2]):
            for iterator2, l in enumerate(locs):
                loc_maxs.append(l)
                snr_maxs_.append(output[:,3][iterator1][iterator2])
        loc_maxs = np.array(loc_maxs)
        snr_maxs_ = np.array(snr_maxs_)


        loc_maxs_ = []
        for i,loc_i in enumerate(loc_maxs):
            # compute differences between this loc_max and all others
            diffs = []
            for j,loc_j in enumerate(loc_maxs):
                if i != j: #and snr_maxs_[j] < 10: # still be sensitive to big dips
                    diffs.append(np.abs(loc_i - loc_j))

            diffs = np.array(diffs)

            #if np.any(diffs < window):

            ########### UNCOMMENT BELOW TO REJECT FPS ######
            if np.any(diffs < 50):
                continue

            # if loc_i found to be isolated, store this value
            else:
                loc_maxs_.append(loc_i)

            #loc_maxs_.append(loc_i)
        loc_maxs_ = np.array(loc_maxs_)
        print('# of Isolated peaks:', loc_maxs_.shape)


        # now reassociate this peaks with their source and sigma value
        peaks = []
        for l_ in loc_maxs_:
            for i, loc_list in enumerate(output[:,2]):
                if l_ in loc_list:
                    peaks.append([output[:,0][i], output[:,1][i][loc_list.index(l_)],
                                  l_, output[:,3][i][loc_list.index(l_)]])
        peaks = np.array(peaks)

        ordered_sim_params = sim_params[sim_params[:,0].argsort()]

        '''
        # load data and generate a matrix of observations
        lc_files = [f for d, s, f in os.walk(dir)][0]

        # order the files
        phot_files = []
        seq_files = []
        sky_files = []
        batch_ids = np.arange(0, 1000).astype(str)
        for b in batch_ids:
            for f in lc_files:
                if 'photometry' in f and 'batch' + b + '.npy' in f:
                    phot_files.append(f)
                elif 'seq' in f and 'batch' + b + '.npy' in f:
                    seq_files.append(f)
                elif 'sky' in f and 'batch' + b + '.npy' in f:
                    sky_files.append(f)

        n = 0 # use as reference for array dimensions
        fs = np.load(os.path.join(dir, phot_files[n])) # use for shape info
        ss = np.load(os.path.join(dir, sky_files[n]))
        data = np.zeros((len(phot_files), fs.shape[0], fs.shape[1])) # data matrix
        t = np.zeros((len(seq_files), fs.shape[0])) # observation sequence numbers (i.e. time)
        bkgs = np.zeros((len(sky_files), ss.shape[0], ss.shape[1]))


        data = batch_files(phot_files, data)
        t = batch_files(seq_files, t)
        bkgs = batch_files(sky_files, bkgs)

        # reshape and change dtype
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]).astype(np.float32)
        bkgs = bkgs.reshape(bkgs.shape[0] * bkgs.shape[1], bkgs.shape[2]).astype(np.float32)
        t = t.flatten().astype(np.float32)

        X = np.copy(data)
        '''
        recovered = []
        missed = []
        for sps in ordered_sim_params:

            # unpack parameters
            lc_id, t0, radius, dist, phi, b = sps

            # generate the curve and compute where the occultation midpoint is
            curve = genCurve(exposure, startLam, endLam, radius, b, dist, angDi, shiftAdj, phi)
            t1 = t0 + len(curve)
            mid_point = int((t1 + t0) / 2)


            t1 = t0 + len(curve)
            lc_id, t0, t1 = int(lc_id), int(t0), int(t1)
            X[:, lc_id][t0:t1] += (np.median(X[:, lc_id]) * (curve - 1))



            # systematic variation is a funtion of flux, so normalise out
            X_norm = X / np.median(X, axis=0)

            # select all stars with rms < 0.1
            comparison_stars = np.where(np.std(X_norm[:, :], axis=0) < 0.1)[0]

            # if one or less comparison stars fail to meet the above condition, just use the stablest 10 OR brightest 10
            #if len(comparison_stars) <= 1:
            if len(comparison_stars) <= 2:
                comparison_stars = np.arange(0, 10, 1) # brightest
            #wav = np.average(X_norm[:, comparison_stars],
            #              weights=np.std(X_norm[:, comparison_stars], axis=0), axis=1)

            comp_stars = [c for c in comparison_stars if c != lc_id]
            #print("Number of comparison stars:", len(comp_stars))
            #print(lc_id)
            #print(comp_stars)
            #wav = np.average(X_norm[:, comparison_stars], weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
            wav = np.average(X_norm[:, comp_stars],
                         #weights=np.sqrt(np.median(X[:, comparison_stars], axis=0)), axis=1)
                         weights=1/np.var(X_norm[:, comp_stars], axis=0), axis=1)

            lc_ = X_norm[:, lc_id].T / wav


            # identify peaks assocaited with this light curve
            peaks_ = peaks[peaks[:,0] == int(lc_id)]

            # is there a peak near where the occultation should be?

            diffs = np.abs(t[peaks_[:,2].astype(int)] - t[mid_point])
            if np.any(diffs < 500):
                print('Found:', sps, 1. / mad_std(lc_))
                recovered.append([lc_id, t0, radius, dist, phi, b, 1. / mad_std(lc_)])

            else:
                print('Missed:', sps, 1. / mad_std(lc_))
                missed.append([lc_id, t0, radius, dist, phi, b, 1. / mad_std(lc_)])
                continue


        recovered = np.array(recovered)
        missed = np.array(missed)

        print('Recovered shape:', recovered.shape)
        print('Missed shape:', missed.shape)

        #np.save('analysisSIM/results/' +  dir + '_recovered_' + str(x) + '_noFPreject.npy', recovered)
        #np.save('analysisSIM/results/' +  dir + '_missed_' + str(x) + '_noFPreject.npy', missed)
        np.save('analysisSIM/results/' +  dir.split('\\')[-1] + '_recovered_' + str(x) + '.npy', recovered)
        np.save('analysisSIM/results/' +  dir.split('\\')[-1] + '_missed_' + str(x) + '.npy', missed)
