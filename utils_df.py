import os
import glob
import numpy as np
from collections import defaultdict
import pandas as pd
from shutil import copyfile
import jinja2
from matviz.matviz import etl
import pickle
import collections
import scipy.signal as signal

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

TEMPLATE_PATH = 'config/template.html'


PATH_FIGURES = './figures'
PATH_RESULTS = './results'
PATH_REPORTS = './reports'


EXAM_LIST = [
    'prosaccade',
    'selfpacedsaccade',
    'vor',
    'smoothpursuit',
    'pupillaryreflex',
    'convergence',
    'smoothpursuit2d',
    'gaze'
]

# The key data columns that matter for sampling rate - we are recording these
KEY_COLUMNS = [
               'Right Pupil Diameter',
               'Left Pupil Diameter',
               'Right Eye Direction Y',
               'Right Eye Direction X',
               'Combine Eye Direction X',
               'Left Eye Direction Y',
               'Left Eye Direction X',
               'Combine Eye Direction Y',
               'Camera Position X'
               ]

SMALL_FIG_LIST =  ['plot_ds']

MULT_FACTOR = 5 # to convert units of eye direction to units on the wall



def get_individual_exam(exam):
    """Get many data frames for a specific exam"""
    # todo
    all_df = []
    return all_df


def robust_get_stimulus(df, xy='X', num='1'):
    """Get the main stimulus whether it is a dot or a sphere"""
    if 'Dot1 Position X' in df:
        return df['Dot' + num + ' Position ' + xy]
    elif 'Sphere Position' in df:
        return df['Sphere Position']
    elif 'Sphere Position ' + xy in df:
        return df['Sphere Position ' + xy]
    elif 'Dot ' + num + ' Position ' + xy in df:
        return df['Dot ' + num + ' Position ' + xy]
    elif 'Stimulus ' + num + ' Position ' + xy in df:
        return df['Stimulus ' + num + ' Position ' + xy]
    elif 'Stimulus ' + num in df:
        return df['Stimulus ' + num]
    else:
        print(df.columns)

        raise Exception("missing or misnamed stimulus column")


def clean_df(df):
    """
    Convert the numerical error values into numpy nans
    Backwards compatibility for data collections with only one pupil recorded
    :param df: timeseries df
    :return: timeseries df
    """

    # todo: check if we still need this step
    # remove the whitespace around column names
    df.columns = df.columns.map(str.strip)

    error_message = ''

    # Check for a weird misformed state where there are double the correct number of columns
    # by checking to see that every column name is unique
    if len(df.columns) != len(np.unique(df.columns)):
        logger.warning("Duplicate columns in df.")
        error_message += 'Duplicate columns in df'
        return  pd.DataFrame(), error_message


    # overwrite the total_time column with time provided by date column
    df['Total Time'] = clean_time(df)


    # clean the data for bad formatting
    clean_xy = lambda x: np.nan if (x == 0 or x == -1) else x
    clean_pupil = lambda x: np.nan if x == -1 else x

    if 'Pupil Diameter' in df:
        df['Pupil Diameter'] = df['Pupil Diameter'].apply(clean_pupil)
        df['Left Pupil Diameter'] = df['Pupil Diameter']
        df['Right Pupil Diameter'] = df['Pupil Diameter']
    else:
        df['Pupil Diameter'] = df['Left Pupil Diameter'].apply(clean_pupil)
        df['Left Pupil Diameter'] = df['Left Pupil Diameter'].apply(clean_pupil)
        df['Right Pupil Diameter'] = df['Right Pupil Diameter'].apply(clean_pupil)

    df['Combine Eye Direction X'] = df['Combine Eye Direction X'].apply(clean_xy)
    df['Combine Eye Direction Y'] = df['Combine Eye Direction Y'].apply(clean_xy)
    df['Combine Eye Position X'] = df['Combine Eye Position X'].apply(clean_xy)
    df['Combine Eye Position Y'] = df['Combine Eye Position Y'].apply(clean_xy)

    df['Left Eye Direction X'] = df['Left Eye Direction X'].apply(clean_xy)
    df['Left Eye Position X'] = df['Left Eye Position X'].apply(clean_xy)
    df['Left Eye Direction Y'] = df['Left Eye Direction Y'].apply(clean_xy)
    df['Left Eye Position Y'] = df['Left Eye Position Y'].apply(clean_xy)

    df['Right Eye Direction X'] = df['Right Eye Direction X'].apply(clean_xy)
    df['Right Eye Position X'] = df['Right Eye Position X'].apply(clean_xy)
    df['Right Eye Direction Y'] = df['Right Eye Direction Y'].apply(clean_xy)
    df['Right Eye Position Y'] = df['Right Eye Position Y'].apply(clean_xy)

    if 'Combine Eye Position X' in df:
        df['Combine Eye Position X'] = df['Combine Eye Position X'].apply(clean_xy)
        df['Combine Eye Position Y'] = df['Combine Eye Position Y'].apply(clean_xy)
        df['Combine Eye Position Z'] = df['Combine Eye Position Z'].apply(clean_xy)



    if 'Instruction Dot Size' in df:
        instructions_dot_time = np.sum(np.logical_not(np.isnan(df['Instruction Dot Size'])))
        if instructions_dot_time == 0:
            logger.warning("Instructions dot exists but is all NaNs.")

    if sufficient_sampling_rate(df):
        df = remove_blinks(df)
    else:
        logger.warning("Sampling rate problem.")
        error_message += 'Sampling rate problem. '

    if too_many_nans(df):
        logger.warning(" Very high number of NaNs.")
        error_message += 'Very high number of NaNs.'

    return df, error_message




def sufficient_sampling_rate(df):
    """
    Checks to see if the sampling rate is normal, directly by looking for duplicate values
    :param df:
    :return: boolean if the sampling rate passes the requirement
    """
    x = 100 * np.arange(len(df) - 1) / len(df)

    # up to this percentage of points can be duplicates
    thresh = 0.95
    cut_idx = int(thresh * len(x))

    for cur_col in KEY_COLUMNS:
        y = np.sort(np.abs(np.diff(df[cur_col])))
        # if the thresh fraction'th point is zero then we have a sampling rate problem
        if y[cut_idx] == 0:
            return False

    return True

def too_many_nans(df):
    """
    Checks to see if the key data we need are nan
    :param df:
    :return: boolean if nan threshold meets the criteria
    """
    # up to this percentage of samples can be nans
    thresh = 0.85
    for cur_col in KEY_COLUMNS:
        percent_nan = np.sum(np.isnan(df[cur_col])) / len(df)
        # if the thresh fraction'th point is zero then we have a sampling rate problem
        if percent_nan > thresh:
            return True

    return False


def flip_x(df, exam_version):
    """
    Identify if the sampling rate is high. Right now exam version is not used
    :param df:
    :return:
    """

    sampling_rate = np.median(df['Delta Time'])

    if sampling_rate < .015:
        df['Combine Eye Direction X'] = -df['Combine Eye Direction X']
        df['Combine Eye Position X'] = -df['Combine Eye Position X']
        df['Left Eye Direction X'] = -df['Left Eye Direction X']
        df['Left Eye Position X'] = -df['Left Eye Position X']
        df['Right Eye Direction X'] = -df['Right Eye Direction X']
        df['Right Eye Position X'] = -df['Right Eye Position X']

    return df

def remove_blinks(df):
    """
    This will remove the x, y and pupil data surrounding a blink for being probably bad
    Details on the heuristc and analysis here:
    https://www.notion.so/reactneuro/Blink-Detection-76d29622b71d43c6938fbce0b7bca6f9

    A blink is detected if:
    There are overlapping X & Y NaN data (after padding) OR pupil NaN Data
    Padding is 4 samples on each side.

    :param df: timeseries df
    :return: timeseries df
    """

    cols_to_nanify = ['Right Pupil Diameter', 'Left Pupil Diameter',
                      'Left Eye Direction X',
                      'Right Eye Direction X',
                      'Left Eye Direction Y',
                      'Right Eye Direction Y',
                      'Combine Eye Direction X',
                      'Combine Eye Direction Y',
                      ]

    if 'Combine Eye Position X' in df:
        cols_to_nanify += ['Combine Eye Position X',
                           'Combine Eye Position Y',
                           'Combine Eye Position Z']

    # If any of these variables are NaN, then we want to exclude all the others too
    master_nan = np.isnan(df[cols_to_nanify]).any(axis=1)

    # Expand the region of nans by NUM_TO_PAD/2 on each side, since bad data is usually near blinks
    NUM_TO_PAD = 8
    master_nan = master_nan.rolling(window=NUM_TO_PAD, center=True).mean().astype(bool)
    # set the values to be np.nan in this territory
    df.loc[df.index[master_nan], cols_to_nanify] = np.nan

    return df




def count_blinks(df):
    """
    Count the number of blinks in the exam.
    This should happen after `remove_blinks` is run
    :param df:
    :return:
    """
    num_blinks = 0
    if 'Pupil Diameter' in df.columns:
        blinks = etl.start_and_ends(np.isnan(df['Pupil Diameter']))
        num_blinks = len(blinks)
    return num_blinks



# for smooth_pursuit2d
def df_to_raw_error(df):
    def error_position(z, stim):
        return z / (stim/np.abs(stim)) - np.abs(stim)

    def error_angle(z, stim):
        return error_position(z, stim).imag

    def error_radius(z, stim):
        return error_position(z, stim).real

    t, z, stim = get_t_z_stim(df)
    z = z + 1j * 0.5
    stim = stim * .0075
    zs = etl.nan_smooth(z,10)
    return error_position(zs, stim)

def df_to_error(df):
    """
    This one returns the error sliding over t
    :param df:
    :return:
    """
    def error_position(z, stim):
        return z / (stim / np.abs(stim)) - np.abs(stim)

    t, z, stim = get_t_z_stim(df)
    z = z + 1j * 0.5
    stim = stim * .0075
    zs = etl.nan_smooth(z,10)
    return error_position(zs, stim) + t - np.nanmin(t)


def df_to_periodogram(df, eye='left', xy='x', type='periodogram'):
    ds = np.nanmedian(df['Delta Time'])

    t, left, right = get_t_lr(df)
    if xy == 'x':
        x_L = left.real
        x_R = right.real
    else:
        x_L = left.imag
        x_R = right.imag
    I = np.logical_not(np.isnan(x_R))

    if eye == 'right':
        x = x_R[I]
    else:
        x = x_L[I]

    t = t[I]
    ti, xi = etl.interp_nans(t, x)
    if type == 'periodogram':
        freq, power = signal.periodogram(xi, 1 / ds, window='hann')
    elif type == 'welch':
        freq, power = signal.welch(xi, 1 / ds, window='hann')
    return freq, power


def get_exam_start(df):
    """Get the start of the exam when the instructions dot goes to zero"""
    if 'Exam Status' in df:
        exam_start_idx = np.argmax((df['Exam Status'] == 'Exam').values)
    elif 'Instruction Dot Size' in df:
            stim = df['Instruction Dot Size']
            exam_start_idx = np.argmax(np.array(np.isnan(stim)))
    else:
        raise Exception('Cannot find exam start: Missing Exam Status - or Instructions dot.')
    exam_start_idx = exam_start_idx + 1 # hack to deal with first two samples not monotically increasing
    exam_start_time = df['Total Time'][exam_start_idx]
    return exam_start_idx, exam_start_time


def get_t_z_stim(df, eye='Combine'):
    """Get stimulus position and data during the experiment only"""
    t, z = get_t_z(df, eye=eye)
    t, stim = get_t_stim(df)
    return t, z, stim


def get_t_z_stims(df, eye='Combine'):
    """Get stimulus position and data during the experiment only"""
    t, z = get_t_z(df, eye=eye)
    t, stims = get_t_stims(df)
    return t, z, stims


def get_t_stim(df):
    """Get the stimulus during the exam"""
    start_idx, _ = get_exam_start(df)
    t = df['Total Time'].values[start_idx:]
    stim_z = robust_get_stimulus(df, 'X').values + 1j * robust_get_stimulus(df, 'Y').values
    stim_z = stim_z[start_idx:]
    return t, stim_z

def get_df_stroop(df):
    """Get the stimulus and other relevant eye data for stroop"""
    df = get_key_data(df)

    key_columns = ['Total Time', 'Active Text', 'Active Color', 'Color Index', 'z', 'pupil']

    start_idx, _ = get_exam_start(df)

    stim_df = df.loc[start_idx:, key_columns]

    return stim_df

def get_df_bostonnaming(df):
    # todo: actually get the columsn that we need --> 'Image'
    """Get the image stimulus and the relevant eye data for boston naming"""
    df = get_key_data(df)

    key_columns = ['Total Time', 'Active Image', 'Image Index', 'z', 'pupil']

    start_idx, _ = get_exam_start(df)

    stim_df = df.loc[start_idx:, key_columns]

    return stim_df

def get_t_stims(df):
    """Get the stimulus during the exam"""
    start_idx, _ = get_exam_start(df)
    t = df['Total Time'].values[start_idx:]
    stims_z = []
    for stim_num in ['1', '2']:
        stim_z = robust_get_stimulus(df, 'X', num=stim_num).values + 1j * robust_get_stimulus(df, 'Y', num=stim_num).values
        stim_z = stim_z[start_idx:]
        stims_z.append(stim_z)
    return t, stims_z

def clean_time(df):
    """
    parse the date out of the string
    convert it to total sweconds

    :param df:
    :return:
    """
    def process_date(w):
        w = w[:-3]
        w = w[:-5] + '.' + w[-4:] # replace the ":" with a "." so it can be interpreted by the parsers
        try:
            w = pd.Timestamp(w)
        except:
            w = None
        return w

    def date_to_seconds(all_dates):
        #     convert the date column into total seconds (move this forward probably maybe?)
        cur_seconds = (all_dates - all_dates.iloc[0]).values.astype('float64') / 1e9
        return cur_seconds

    all_dates = df['Date'].apply(process_date)
    t = date_to_seconds(all_dates)

    return t


def get_t_z(df, eye='combine'):
    """Get time and x/y data (as complex z) during the exam"""
    start_idx, _ = get_exam_start(df)

    # parse out which eye to get data for
    eye_dict = {
                'left': "Left",
                'right': "Right",
                'combine': "Combine"
    }
    cur_eye = eye_dict[eye.lower()]

    t = df['Total Time'].values[start_idx:]
    x = df[cur_eye + ' Eye Direction X'].values[start_idx:]
    y = df[cur_eye + ' Eye Direction Y'].values[start_idx:]

    return t, MULT_FACTOR *(x + 1j * y)

def get_z(df, eye='combine'):

    # parse out which eye to get data for
    eye_dict = {
                'left': "Left",
                'right': "Right",
                'combine': "Combine"
    }
    cur_eye = eye_dict[eye.lower()]

    x = df[cur_eye + ' Eye Direction X'].values
    y = df[cur_eye + ' Eye Direction Y'].values

    return MULT_FACTOR *(x + 1j * y)



def get_t_lr(df):
    """Get the left and right eye data, x/y stored as complex z"""
    t, left = get_t_z(df, eye='left')
    t, right = get_t_z(df, eye='right')

    return t, left, right


def get_t_pupil(df):
    """Get the pupil diameter during the exam"""
    start_idx, _ = get_exam_start(df)

    t, pupil, stim = get_t_pupilz_stim(df)

    stim_idx_tmp = etl.start_and_ends(stim)
    if len(stim_idx_tmp) > 0:
        stim_idx = stim_idx_tmp[0][0]
        stim_t = t[stim_idx]
        I = (stim_t - 11 < t) & (t < stim_t + 8)

        pupil = pupil[I]
        stim = stim[I]
        t = t[I]
    else:
        logger.warning("Could not find stimulus")
        t = np.array([])
        pupil = np.array([])
        stim = np.array([])

    return t, pupil, stim

def get_t_xy(df):
    """NOTE: using X and Y is depreciated now that we've graduated to complex representations"""
    start_idx, _ = get_exam_start(df)
    t = df['Total Time'].values[start_idx:]
    x = df['Combine Eye Direction X'].values[start_idx:]
    y = df['Combine Eye Direction Y'].values[start_idx:]
    return t, x, y


def get_inst_xy_t_stim(df):
    """Get the x,y,t,stime for the instructions period pre-exam"""
    start_idx, start_time = get_exam_start(df)
    t = df['Total Time'].values[:start_idx]
    x = df['Combine Eye Direction X'].values[:start_idx]
    y = df['Combine Eye Direction Y'].values[:start_idx]
    stim = df['Instruction Dot Size'].values[:start_idx]
    return x, y, t, stim


def get_xy_0(df):
    """calculate the inital x and y 'zero' points where the insturctions dot is located"""
    start_idx, start_time = get_exam_start(df)
    x = df['Combine Eye Direction X'].values[:start_idx]
    y = df['Combine Eye Direction Y'].values[:start_idx]
    stim = df['Instruction Dot Size'].values[:start_idx]
    I = np.diff(stim) < 0
    x_0 = np.median(x[1:][I])
    y_0 = np.median(y[1:][I])
    return x_0, y_0




def get_t_pupilz_stim(df):
    """Get the pupil diameter during the exam"""
    start_idx, _ = get_exam_start(df)

    t = df['Total Time'].values[start_idx:]

    pupil_R = df['Right Pupil Diameter'].values[start_idx:]
    pupil_L = df['Left Pupil Diameter'].values[start_idx:]
    pupil = pupil_R + 1j * pupil_L
    stim = df['Stimulus 1'].values[start_idx:]
    return t, pupil, stim


def get_t_pupilz(df):
    """Get the pupil diameter during the exam"""
    start_idx, _ = get_exam_start(df)

    t = df['Total Time'].values[start_idx:]

    pupil_R = df['Right Pupil Diameter'].values[start_idx:]
    pupil_L = df['Left Pupil Diameter'].values[start_idx:]
    pupil = pupil_R + 1j * pupil_L
    return t, pupil


def get_key_data(df):
    """
    Get the various data:
    pupil diameter during the exam, z
    eye movements (z)
    """

    pupil_R = df['Right Pupil Diameter'].values
    pupil_L = df['Left Pupil Diameter'].values
    df['pupil'] = pupil_R + 1j * pupil_L

    df['z'] = get_z(df)
    df['zL'] = get_z(df, eye='left')
    df['zR'] = get_z(df, eye='right')

    return df





