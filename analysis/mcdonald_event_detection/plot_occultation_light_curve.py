# imports
from math import hypot, ceil
import numpy as np
from math import pi, cos, tan
from scipy.special import jv  # Bessel function
from itertools import product
import matplotlib.pyplot as plt


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

#exposure = 0.0012
#startLam = 6e-7 # 4e-7 start of wavelength range
#endLam = 6e-7 # end of wavelength range
#objectRad = 500 # object radius, m
#impact = 1000 # impact parameter, m
#dist = 40 # object distance, AU
#angDi = 0.02 # angular diameter of star, mas
#shiftAdj = 0
#phi = 0

startLam = 2e-7 # 4e-7 start of wavelength range[m]
endLam = 11e-7 # end of wavelength range [m]
impact = 0 # impact parameter [m]
shiftAdj = 0 # don't touch
phi = 0 # opposition angle, degrees

# input parameters from the terminal
exposure = input('Camera exposure time [ms]:')
exposure = float(exposure) / 1e3 # ms -> s

objectRad = input('Object radius [km]:')
objectRad = float(objectRad) * 1e3 # km -> m

dist = input('Object distance [Au]:')
dist = float(dist)

#impact = input('Impact parameter [km]:')
#impact = int(impact) * 1e3 # km -> m

angDi = input('Stellar angular diameter [mas]:')
angDi = float(angDi)

# compute occultation curve
print('Computing occultation light curve....')
curve = genCurve(exposure, startLam, endLam, objectRad,
                 impact, dist, angDi, shiftAdj, phi)
print('Done!')
# plot
t = exposure * np.arange(0, len(curve))
t -= np.median(t)
plt.plot(t, curve)
plt.xlim(-1,1) # time interval (-1 second to  +1 second)
plt.ylim(0, 1.5)
plt.title('Sampling=%.2f [Hz], Object radius [km]=%.2f, Object distance [Au]=%d, $\Theta_*=%.2f$ [mas]' % (1/exposure, objectRad/1e3, dist, angDi))
plt.xlabel('$t$ [seconds]')
plt.ylabel('$I(t)$')
plt.grid()
plt.show();
