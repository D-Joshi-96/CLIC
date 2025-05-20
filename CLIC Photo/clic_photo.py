import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import scipy.signal as sig
from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy import trapz as trapz

def mad(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        mad = 1e-12

    return mad

def mono_exponential(t, a, b, c):
    return a * np.exp(-b * t) + c

def bleach_correction(t, y):
    # Fit the exponential decay model
    popt, _ = curve_fit(mono_exponential, t, y, p0=(1, 0.001, 0))
    # Compute fitted baseline
    baseline = mono_exponential(t, *popt)
    # Detrend the signal
    bleach_corrected = y - baseline

    return bleach_corrected

def process_rawdata(filepath, signalcol, timecol, iso_flag, gcamp_flag, session_length, artifact_threshold, gaussian_sigma, rolling_window, bleach_correction = False):

    """Takes the interleaved photometry CSV file and produces a master dataframe with the raw signal corrected for bleaching, filtered for artifacts and then converted to deltaF/F. The zscore is calculcated over a moving window passed over the deltaF/F."""

    # importing the raw csv files into pandas dataframes for each signal channel i.e. isosbestic and Ca-dependent signal from GCaMP

    df = pd.read_csv(filepath)

    iso_df = df[df["LedState"] == iso_flag]
    gcamp_df = df[df["LedState"] == gcamp_flag]

    iso_df = iso_df.reset_index(drop = True)
    gcamp_df = gcamp_df.reset_index(drop = True)

    iso_df["iso time"] = iso_df[timecol] - iso_df[timecol].min()
    gcamp_df["gcamp time"] = gcamp_df[timecol] - gcamp_df[timecol].min()
    
    iso_df = iso_df[["iso time", "LedState", signalcol]]
    iso_df = iso_df.rename(columns={signalcol:"iso signal raw"})
    gcamp_df = gcamp_df[["gcamp time", "LedState", signalcol]]
    gcamp_df = gcamp_df.rename(columns={signalcol:"gcamp signal raw"})

    iso_df = iso_df.loc[iso_df["iso time"] <= session_length]
    gcamp_df = gcamp_df[gcamp_df["gcamp time"] <= session_length]

    min_length = min(len(iso_df), len(gcamp_df))
                     
    iso_df = iso_df.iloc[:min_length]
    gcamp_df = gcamp_df.iloc[:min_length]

    # filtering artifacts based on a threshold of median absolute deviations of the raw data

    iso_median = np.median(iso_df["iso signal raw"])
    iso_mad = np.median(np.abs(iso_df['iso signal raw'] - iso_median))

    iso_artifacts = iso_df["iso signal raw"] > iso_median + (artifact_threshold * iso_mad)
    if bleach_correction == True:
        iso_df["iso signal"] = bleach_correction(iso_df["iso time"], iso_df["iso signal raw"]) #creating a copy of the raw signal which is detrended and from which artifacts will be filtered
    else:
        iso_df["iso signal"] = iso_df["iso signal raw"]
    iso_df.loc[iso_artifacts, 'iso signal'] = np.nan  # Mark artifacts as NaN in the "signal" column at the index where the value exceeds the threshold

    iso_not_nan = ~np.isnan(iso_df["iso signal"])
    iso_df["iso signal"] = np.interp(np.arange(iso_df["iso signal"].shape[0]),
                                     np.arange(iso_df["iso signal"].shape[0])[iso_not_nan], 
                                     iso_df["iso signal"][iso_not_nan])

    gcamp_median = np.median(gcamp_df["gcamp signal raw"])
    gcamp_mad = np.median(np.abs(gcamp_df['gcamp signal raw'] - gcamp_median))

    gcamp_artifacts = gcamp_df["gcamp signal raw"] > gcamp_median + (artifact_threshold * gcamp_mad)
    if bleach_correction == True:
        gcamp_df["gcamp signal"] = bleach_correction(gcamp_df["gcamp time"], iso_df["gcamp signal raw"]) #creating a copy of the raw signal which is detrended and from which artifacts will be filtered
    else:
        gcamp_df["gcamp signal"] = gcamp_df["gcamp signal raw"]
    gcamp_df.loc[gcamp_artifacts, 'gcamp signal'] = np.nan  # Mark artifacts as NaN at the index where the value exceeds the threshold

    gcamp_not_nan = ~np.isnan(gcamp_df["gcamp signal"])
    gcamp_df["gcamp signal"] = np.interp(np.arange(gcamp_df["gcamp signal"].shape[0]),
                                         np.arange(gcamp_df["gcamp signal"].shape[0])[gcamp_not_nan],
                                         gcamp_df["gcamp signal"][gcamp_not_nan]) 

    # Calculating dF/F with a gaussian filter applied for smoothing

    iso = iso_df["iso signal"]
    ca = gcamp_df["gcamp signal"]

    slope, intercept, r_value, p_value, std_err = linregress(x=iso, y=ca)
    ca_scaled = intercept + slope * iso

    dff = (ca - ca_scaled)/ca_scaled

    #dff = butter(2, 4, btype='low', fs=fps)
    #dff = filtfilt(b,a, dff)

    dff = gaussian_filter1d(dff, sigma = gaussian_sigma)

    gcamp_df["gcamp dff"] = dff

    # Calculate a rolling median and median absolute deviation to evaluate a modified zscore

    gcamp_df['gcamp_moving_median'] = gcamp_df['gcamp dff'].rolling(window=rolling_window, center=True).median()
    gcamp_df['gcamp_moving_mad'] = gcamp_df['gcamp dff'].rolling(window=rolling_window, center=True).apply(mad, raw=True)

    # Calculate moving modified z-score based on median and MAD as opposed to mean and STD
    gcamp_df['gcamp zscore'] = (0.6745*(gcamp_df['gcamp dff'] - gcamp_df['gcamp_moving_median'])) / gcamp_df['gcamp_moving_mad']

    # combining the iso and gcamp dataframes into a master dataframe 
    return_df = pd.concat([iso_df[["iso time", "iso signal raw", "iso signal"]],
                           gcamp_df[["gcamp time", "gcamp signal raw", "gcamp signal", "gcamp dff", "gcamp_moving_median", "gcamp_moving_mad", "gcamp zscore"]]], axis=1)
    
    return_df = return_df[return_df["gcamp zscore"].notna()]

    return return_df

def add_bins(input_df, bin_size):
    bins = np.arange(0, 36000, bin_size) #arguments to define a bin = start, end + 1 bin, bin size - all in seconds
    labels = [i +1 for i,j in enumerate(bins)]
    input_df["bin"] = pd.cut(input_df["iso time"], bins, labels = labels[:-1], right = False)
    input_df['bin'] = input_df['bin'].astype(int)


def peaky_finder(input_df, spike_threshold, signal = "gcamp"):

    """Takes the master dataframe of the dff and zscores to evaluate spikes and waves that occur around each spike."""
    
    df = input_df.copy()

    # Detecting Spikes

    spike_timestamps = [] #initialise an empty list that will be populated with the timestamps of spikes

    # Create a threshold for detecting spikes based on a median absolute deviation from the median of the gross z-score
    median = np.median(df[signal + " zscore"])
    mad = np.median(np.abs(df[signal + " zscore"] - median))
    threshold = median + (spike_threshold * mad)
    
    # Identify clusters of regions where the z-score crosses the threshold and find the peak within it (i.e. the highest z-score within the region)
    above_thresh = df[signal + " zscore"] > threshold
    spike_regions = (above_thresh != above_thresh.shift()).cumsum()
    df['spike_group_raw'] = np.where(above_thresh, spike_regions, np.nan)
    df['spike_group'] = (df['spike_group_raw'].dropna().astype('category').cat.codes)

    # Get the timestamps of the max zscore per spike group which are the regions where the z-score was crossing the spike threshold
    spike_peaks = (df.dropna(subset=['spike_group']).groupby('spike_group').apply(lambda g: g.loc[g[signal + " zscore"].idxmax()], include_groups = False))
    spike_timestamps.extend(spike_peaks[signal + " time"].values.tolist())    
    spike_timestamps = sorted(spike_timestamps)

    #create a column called "spike" in the input_df and score it with 1 where the spike was detected and a 0 everywhere else
    df['spike'] = df[signal + " time"].isin(spike_timestamps).astype(int)

    del df['spike_group_raw']
    df["spike count"] = df["spike_group"] + 1

    # Calculating AUCs under, and the duration of, the waves that are made up by areas of spiky activity

    spike_idx = df[df["spike"]==1].index.to_list()
    wave_start, wave_end = [], []

    for i in spike_idx:
        while df.loc[i, signal + " zscore"] > 0:
            i = i-1
        wave_start.append(i)

    for i in spike_idx:
        while df.loc[i, signal + " zscore"] > 0 and i < df.index[-1]:
            i = i+1
        wave_end.append(i)
    

    for i,j,k in zip(wave_start, spike_idx, wave_end):

        df.loc[j, "wave start"] = df.loc[i, signal + " time"]
        df.loc[j, "wave end"] = df.loc[k, signal + " time"]
        df.loc[j, "wave auc"] = trapz(df.loc[i:k, signal + " zscore"])

    df["wave duration"] = df["wave end"] - df["wave start"]

    for i in df["wave start"].dropna().unique():
        wave_peak = df.loc[df["wave start"]==i][signal + " zscore"].max()
        df.loc[df["wave start"]==i, "wave peak"] = wave_peak

    #adding a counter for every wave that is detected

    for i,j in enumerate(df["wave start"].dropna().unique()):
        df.loc[df["wave start"]==j, "wave count"] = i + 1

    spike_df = df[df["spike"]==1].reset_index(drop = True)

    wave_df = df[['bin', 'wave start', 'wave end', 'wave auc', 'wave duration', 'wave count', 'wave peak']].groupby("wave count", as_index=False).mean()

    #building a custom bin dataframe

    bin_dict = {}
    bin_dict["bin"], bin_dict["spike count"], bin_dict["avg spike peak"], bin_dict["total wave auc"], bin_dict["avg wave duration"] = [],[],[],[],[]

    for i in df["bin"].unique():
        bin_dict["bin"].append(i)
        bin_dict["spike count"].append(len(df.loc[(df["bin"]==i) & (df["spike"]==1)]))
        bin_dict["avg spike peak"].append(np.mean(df.loc[(df["bin"]==i) & (df["spike"]==1)][signal + " zscore"]))
        bin_dict["total wave auc"].append(sum(wave_df.loc[wave_df["bin"]==i]["wave auc"]))
        bin_dict["avg wave duration"].append(np.mean(wave_df.loc[wave_df["bin"]==i]["wave duration"]))

    bin_df = pd.DataFrame(bin_dict).fillna(0)

    return df, spike_df[["spike count", signal + " time", "bin", signal + " zscore"]], wave_df, bin_df

def find_closest_index(df_column, value):
    return (df_column - value).abs().idxmin()

def gen_psth(fp_df, event_file_path, event_col, event_timecol, snip_window, baseline_zscore = False, baseline_window = None):

    """Takes the dataframe from the master dataframe created from the process_rawdata() function and snips the deltaF/F around an event from interest specified from the csv file with timestamps of behavioural events."""

    fp_df = fp_df.copy().reset_index(drop = True)
    
    event_counter = 0
    event, snip_t, gcamp_zscore = [], [], []
    
    fps = round(np.mean(1/np.diff(fp_df["iso time"])))

    event_df = pd.read_csv(event_file_path)
    event_df["time"] = event_df[event_timecol] - event_df[event_timecol].min()
    event_df[event_col + " on"] = (event_df[event_col] & ~event_df[event_col].shift(1, fill_value = False)).astype(int)
    
    for i in event_df[event_df[event_col + " on"]==1]["time"].values.tolist():
        
        if i in fp_df["gcamp time"].values:
            index = fp_df[fp_df["gcamp time"] == i].index[0]
        else:
            index = find_closest_index(fp_df["gcamp time"], i)
        gcamp_signal = fp_df.iloc[index-(snip_window[0]*fps):index+(snip_window[1]*fps)+1]["gcamp dff"].values.tolist()
        
        for j in gcamp_signal:
            if baseline_zscore == True and baseline_window != None:
                snip_baseline = fp_df.iloc[index-(baseline_window[0]*fps):index+(baseline_window[1]*fps)+1]["gcamp dff"].values.tolist()
                gcamp_zscore.append((0.6745*(j - np.median(snip_baseline)))/mad(snip_baseline))

            else:
                gcamp_zscore.append((0.6745*(j - np.median(gcamp_signal)))/mad(gcamp_signal))
            event.append(event_counter+1)
            
        for l in [round(x, 2) for x in np.arange(-snip_window[0], snip_window[1] + (1/fps), 1/fps)]:
            snip_t.append(l)

        event_counter = event_counter + 1
        
    snip_df = pd.DataFrame([event, snip_t, gcamp_zscore]).T
    snip_df.columns = ["event_no", "time", "gcamp zscore"]

    return snip_df