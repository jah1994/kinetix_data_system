# imports
from tqdm import tqdm # progressbar
import os
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import time

# Sensitivy or Speed mode
MODE = 'Sensitivity'

# search region is loc +- window
# window is approx 6.25 seconds
if MODE == 'Sensitivity':
    window = 500
elif MODE == 'Speed':
    window = 1875

# miminum comparison star snr
comp_snr_thresh = 10

# skip light curves whose SNR is less than this (where there's no good chance of reliably recovering occultations)
snr_skip_thresh = 5

# matched-filtering snr thresh for candidate occultation events
candidate_snr_thresh = 5

# where to save the results of the analysis
OUT_PATH = r"D:\McDonaldObs_Feb_2024\Analysis\Analysis_18th_Feb"

# if OUT_PATH doesn't exist, make it
if os.path.exists(OUT_PATH) == False:
    os.makedirs(OUT_PATH)


# make directory for the result .npy files if it doesn't exist
#result_dir = os.path.join(OUT_PATH, 'Analysis', 'Analysis_16th_Feb')
#if os.path.exists(result_dir) == False:
#    os.makedirs(result_dir)


# identify subdirectories containg the results of each run
path_to_data = "D:\McDonaldObs_Feb_2024\Results_18th_Feb"
subdirs = [d for d, s, f in os.walk(path_to_data) if 'photometry' in d]

# filter directories to the specified camera mode
subdirs_ = []
for s in subdirs:
    with open(os.path.join(s.split('photometry')[0], 'housekeeping.log'), 'r') as log_file:
        if MODE in log_file.read():
            subdirs_.append(s)
subdirs = subdirs_

print('Data directories to be analysed:')
print(subdirs)

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


# determine sources > limiting snr
def high_snr_sources(X, snr_thresh):

    X_norm = X / np.median(X, axis=0)
    comparison_stars = np.where(np.std(X_norm[:, :], axis=0) < 0.1)[0]

    # if one or less comparison stars fail to meet the above condition, just use the stablest 10 OR brightest 10
    if len(comparison_stars) <= 1:
        #comparison_stars = np.argsort(np.std(X_norm[:, :], axis=0))[:10] # stablest... not always a good choice
        comparison_stars = np.arange(0, 10, 1) # brightest
    wav = np.average(X_norm[:, comparison_stars],
                 weights=1/np.var(X_norm[:, comparison_stars], axis=0), axis=1)

    sources = []
    for i, lc in tqdm(enumerate(X_norm[:, :].T)):

        # detrend data
        lc_ = lc / wav

        # skip lc if low snr some threshold
        if  1. / mad_std(lc_) < snr_thresh:
            continue
        else:
            sources.append(i)

    return sources


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


# cycle over each results subdirectory
for s in subdirs:
    print('Subdirectory:', s)
    # make a subdirectory for the figures if it doesn't exist
    # if the subdirectory is found, the analysis is assumed to have been already done
    try:
        os.makedirs(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'figures'), exist_ok=True)
        os.makedirs(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'closeups'), exist_ok=True)
        os.makedirs(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'results'), exist_ok=True)
    except FileExistsError:
        print('Results directory already exists... skipping')
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

    # identify candidate comparison stars
    comparison_stars = high_snr_sources(X, snr_thresh=comp_snr_thresh)
    if len(comparison_stars) <= 2:
        print('Only %d comparison stars met the SNR condition... defaulting to the 10 brightest sources' % len(comparison_stars))
        comparison_stars = np.arange(0, 10, 1) # brightest

    print("Number of potential comparison stars:", len(comparison_stars))
    print('Comparison stars:', comparison_stars)

    # plot of std vs median flux
    plt.figure(figsize=(10,7))
    plt.scatter(np.median(X, axis=0), np.std(X, axis=0)[:, None], alpha=0.25, c='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log Flux [ADU]')
    plt.ylabel('log rms [ADU]')
    plt.grid()
    #plt.savefig(os.path.join(OUT_PATH, 'analysis/figures/' + str(s.split("\\")[-1]) + '/rms.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'figures', 'rms.png'), bbox_inches='tight')
    plt.close()

    # systematic variation is a funtion of flux, so normalise out
    X_norm = X / np.median(X, axis=0)

    # plot of nomralised fluxes for the comparison stars
    plt.figure(figsize=(10,5))
    plt.plot(t, X_norm[:, comparison_stars], c='black', alpha=0.1)
    plt.ylabel('Relative intensity')
    plt.xlabel('Exposure number')
    plt.grid()
    plt.ylim(-0.3, 1.5)
    #plt.savefig(os.path.join(OUT_PATH, 'analysis/figures/' + str(s.split("\\")[-1]) + '/norm.png'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'figures', 'norm.png'), bbox_inches='tight')
    plt.close()


    ## generate a template bank of candidate occultation signals
    print('Generating template bank...')
    if MODE == 'Sensitivity':
        temp_bank, temp_params = gen_template_bank(0.1, 1, 5, 505, 0.1, 10)
    elif MODE == 'Speed':
        temp_bank, temp_params = gen_template_bank(0.1, 1, 5, 2005, 0.1, 20)

    print("Template bank size:", len(temp_bank))

    ## cycle over each light curve and identifying similarity of each template
    res = [] # stick full results in this
    res_local_peaks = [] # fill this with local peaks for each source
    for i, lc in tqdm(enumerate(X_norm[:, :].T)):

        # ensure that the target source is not included in the comparison star list
        comp_stars = [c for c in comparison_stars if c != i]
        wav = np.average(X_norm[:, comp_stars], weights=1/np.var(X_norm[:, comp_stars], axis=0), axis=1)

        # detrend data
        lc_ = lc / wav

        # skip lc if noise exceeds some threshold
        if 1. / mad_std(lc_) < snr_skip_thresh:
            continue

        plt.figure(figsize=(10,5))
        plt.plot(t, lc, c='black', alpha=0.1, label='Raw')
        plt.plot(t, lc_, c='black', alpha=0.5, label='Detrended')
        plt.ylabel('Relative intensity')
        plt.xlabel('Exposure number')
        plt.title('Source: %d, SNR: %.2f' % (i, 1. / mad_std(lc_)))
        plt.legend()
        plt.grid()
        plt.ylim(1 - 10*mad_std(lc_), 1 + 10*mad_std(lc_))
        #plt.savefig(os.path.join(OUT_PATH, 'analysis/figures/' + str(s.split("\\")[-1]) + '/' + str(i) + '.png'))
        plt.savefig(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'figures', str(i) + '.png'), bbox_inches='tight')
        plt.close()

        print('Starting matched-filtering...')
        tmf = time.time()
        # roll out the template signals...
        for j, tmplt in enumerate(temp_bank):
            flag_occultation_candidate(res, i, j, lc_, tmplt, thresh=candidate_snr_thresh)
        print('Finished matched-filtering in %.2f seconds.' % (time.time() - tmf))

        # help prevent doubling up of identical events
        locs = []

        # if no peaks found, skip
        if len(res) == 0:
            continue

        # restrict results to the current source
        try:
            res_ = np.array(res, dtype=object)
            res_ = res_[res_[:,0] == i]
        except Exception as e:
            print(e)
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
        snrs = np.array(snrs)


        snr_maxs = []
        loc_check = []
        t_ids = []


        # divide time series into window sized grid in which to search for peaks
        grid = np.arange(0, X.shape[0], window) + (window/2)
        grid = grid.astype(int)

        # cycle over the gridded light curve, looking for matched-filtered peaks in each window
        for uloc in grid:

            try:
                snr_max = np.max(snrs[np.where(np.abs(locs - uloc) < (window/2))])
            except ValueError: # zero-sized array if no peaks found in window
                continue
            loc_max = locs[np.where(snrs == snr_max)][0]

            # Already seen this candidate occultation?
            if snr_max in snr_maxs:
                continue

            # Does the SNR peak pass the threshold?
            if snr_max < candidate_snr_thresh:
                continue

            # find the template associated with this particular snr peak
            for sp,snrp in enumerate(snr_peaks):
                if snr_max in snrp:
                    template_id = template_ids[sp]


            # record local dip info
            snr_maxs.append(snr_max)
            loc_check.append(loc_max)
            t_ids.append(template_id)

            ######## plotting #########
            # window for plotting
            n1, n2 = int(loc_max - (window/2)), int(loc_max + (window/2))

            # comparison stars
            if i == 0:
                comp = 1
            else:
                comp = 0

            # light curve median absolute deviation
            mstd = mad_std(lc_)

            if len(lc_) == 0:
                continue

            if len(t[n1:n2]) == 0:
                continue

            curve = temp_bank[template_id]
            plt.figure(figsize=(10,7))
            plt.plot(t[n1:n2],  lc_[n1:n2])
            plt.scatter(t[n1:n2], lc_[n1:n2])
            plt.plot(t[n1:n2], X_norm[n1:n2, comp].T / wav[n1:n2])

            plt.ylim(np.median(lc_) - 10*mstd, np.median(lc_) + 10*mstd)
            plt.grid()
            plt.title('Source:%d, temp_params:%s, $\sigma_{SNR}=%.3f$' % (i, str(temp_params[template_id]),
                                                                       snr_max))
            plt.xlabel('$t$ [Exposure number]')
            plt.ylabel('$I(t)$')

            #plt.savefig(os.path.join(OUT_PATH, 'analysis/closeups/' + str(s.split("\\")[-1]) + '/' + str(i) + '_' + str(loc_max) + '.png'))
            plt.savefig(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'closeups', str(i) + '_' + str(loc_max) + '.png'))
            plt.close()

        res_local_peaks.append([i, snr_maxs, loc_check, t_ids])

    # convert results to an array type
    res = np.array(res, dtype=object)
    print('Results array shape:', res.shape)
    #np.save(os.path.join(result_dir, str(s.split("\\")[-1]) + '.npy'), res)
    np.save(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'results', 'res.npy'), res)

    # and also save the peak information for quick referencing
    res_local_peaks = np.array(res_local_peaks, dtype=object)
    print('Local peak results array shape:', res_local_peaks.shape)
    #np.save(os.path.join(result_dir, str(s.split("\\")[-1])  + '_peaks.npy'), res_local_peaks)
    np.save(os.path.join(OUT_PATH, s.split("\\")[3], s.split("\\")[4], 'analysis', 'results', 'res_local_peaks.npy'), res_local_peaks)
