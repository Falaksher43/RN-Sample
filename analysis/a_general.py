import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as signal
# import logging
import more_itertools
import copy

from matviz.matviz import viz
from matviz.matviz import etl

import utils_df
import saccades
from utils_audio import *

import requests
from spacy.tokens import Token, Span
from spacy.matcher import PhraseMatcher

from sklearn.decomposition import PCA
from pyquaternion import Quaternion

import spacy
import en_core_web_sm
import inflect

import matplotlib
from Levenshtein import distance as levenshtein_distance

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

def chunk_2_lag(cur_chunk, dt):
    """take two signals and compute the lag with peak correlation"""
    eye_X, stim_X = np.transpose(cur_chunk)
    [corrs, lags] = etl.xcorr(eye_X, stim_X, dt)
    return lags[np.argmax(corrs)]


def calibrate_stim(stim):
    return stim * .0085


def error_position(z, stim):
    return z / (stim / np.abs(stim)) - np.abs(stim)


def error_angle(z, stim):
    return error_position(z, stim).imag


def error_radius(z, stim):
    return error_position(z, stim).real


def transform_t_z_stim(t, z, stim):
    """
    Pull out the t,z,stim data
    set t to start at 0,
    smooth the z,
    calibrate the stim to match the eyes
    set all variables to only be where the stim is not nan
    """

    t = t - np.min(t)

    zs = etl.nan_smooth(z, 5)

    # zs = zs * -1
    # zs = np.abs(zs) * np.exp(-np.angle(zs) * 1j)

    stim = calibrate_stim(stim)

    # calculate and return when the stimulus is not nan
    idx_min = list(np.logical_not(np.isnan(stim))).index(True)
    idx_max = list(np.flip(np.logical_not(np.isnan(stim)))).index(True)

    return t[idx_min:-idx_max], zs[idx_min:-idx_max], stim[idx_min:-idx_max]


def error_position(z, stim):
    return z / (stim / np.abs(stim)) - np.abs(stim)


def error_angle(z, stim):
    return error_position(z, stim).imag


def error_radius(z, stim):
    return error_position(z, stim).real


def get_next_exam_param(cur_exam_param, exam):
    """
    Read the
    :param cur_exam_param: the parameter for the current exam being run
    :param exam: the current exam
    :return: the parameter to be used in the next exam
    """
    exam_config_list = etl.load_json("config/next_exam.json")[exam]

    if cur_exam_param in exam_config_list:
        cur_index = exam_config_list.index(cur_exam_param)
        next_index = (cur_index + 1) % len(exam_config_list)
    else:
        raise Exception(str(cur_exam_param) + " not found in config list for " + exam)

    next_exam_param = exam_config_list[next_index]

    return next_exam_param


def convert_to_librosa(audio, sr):
    """
    :param audio: audiosegment object
    :param sample_rate: sample_rate of original file
    :return: resampled librosa ndarray
    """

    samples = audio.get_array_of_samples()
    data = np.array(samples).astype(np.float32) / 32768  # 16 bit
    data = librosa.core.resample(data, audio.frame_rate, sr, res_type='kaiser_best')

    return data


def process_audio_w_chirp(audio, chirp_path='config/chirp_bounded2.wav'):
    sr = audio.frame_rate
    audio_lib = convert_to_librosa(audio, sr=sr)
    chirp, chirp_sr = librosa.load(chirp_path, sr=sr)

    # todo: comment why these numbers
    hp_audio = pass_filter(audio_lib, sr, filter_type='highpass', cutoff=18000)
    hp_sync = pass_filter(chirp, sr, filter_type='highpass', cutoff=18000)
    bp_audio = pass_filter(audio_lib, sr, filter_type='bandpass', cutoff=[125, 8000])

    return hp_audio, hp_sync, bp_audio


def get_peak_corr(signal1, signal2, sr):
    corr, lags = etl.xcorr(signal1, signal2, ds=1 / sr)

    ds = 1 / sr
    ts = lags + len(signal1) * ds - len(signal2) * ds
    threshold = np.max(corr) / 2

    peaks = signal.find_peaks(corr, threshold=threshold, distance=sr)
    peak_locations = peaks[0] / sr - len(signal2) * ds

    return corr, ts, peak_locations


def threshold_passed(audio, log_threshold):
    passed_idx = np.argwhere(np.log(audio ** 2) > log_threshold)
    return passed_idx


def meets_criteria(x, criteria):
    if x in criteria:
        return True
    else:
        return False


def process_instructions(df):
    # start_idx, start_time = hlp.get_exam_start(df)
    x, y, t, stim = utils_df.get_inst_xy_t_stim(df)
    # find the first moment when the stimulus started shrinking
    stim_shrink = np.argmax(stim < 140)

    results = {
        'x_std': np.nanstd(x[stim_shrink:-1]),
        'y_std': np.nanstd(y[stim_shrink:-1]),
        'delta_t': t[-1] - t[stim_shrink],
    }

    return results


def df_to_saccades(df):
    t = df['Total Time'].values
    x = df['Combine Eye Direction Y'].values
    y = df['Combine Eye Direction X'].values
    return saccades.get_saccades(x, y, t)


def get_stim_moves(stim):
    # find if the stimulus moved in the or y direction
    stim_moves = np.abs(np.diff(stim))
    moves = np.nan_to_num(stim_moves).astype(bool)

    # find the moments when the stimulus changes
    move_idxs = np.arange(len(moves))[moves]

    return move_idxs


def process_gaze(time_df, audio_dict):
    t, z, stim = utils_df.get_t_z_stim(time_df)

    results = {'t': t,
               'z': z,
               'stim': stim}

    saccades_df = saccades.cget_saccades(z, t)

    # filter out saccades that are too small
    saccades_df = saccades_df[saccades_df['distances'] > .1]

    # todo: populate gaze metrics dict with values for summary report
    metrics = {'next_exam_params': None}

    return {
        'eye': pd.DataFrame(results),
        'saccades': saccades_df,
        'metrics': metrics,
        'has_error': False
    }


def process_summary(df):
    """Get the same general metrics for each bit"""
    total_times = df['Total Time'].values

    results = dict()
    results['Sampling rate median'] = np.median(df['Delta Time'])
    results['Sampling rate max'] = np.max(df['Delta Time'])
    results['Sampling rate min'] = np.min(df['Delta Time'])
    results['Number of samples'] = len(df['Delta Time'])

    results['Instructions'] = process_instructions(df)

    start_idx, start_time = utils_df.get_exam_start(df)

    results['Total exam duration'] = total_times[-1] - total_times[0]
    results['Exam duration'] = total_times[-1] - start_time

    return results


def pluralize_list_of_words(read_path, write_path):
    '''
    Helper function to create plural forms of words, such as
    those used in the category fluency digital exam
    '''

    p = inflect.engine()

    with open(read_path) as f:
        singular_words = f.read().splitlines()
        plural_words = [p.plural(word) for word in singular_words if not p.singular_noun(word)]
        all_words = singular_words + plural_words

    with open(write_path, 'a') as f:
        unique_words = set(all_words)
        sorted_words = sorted(unique_words)
        sorted_words = (word.lower() for word in sorted_words)
        for word in sorted_words:
            f.write('\n%s' % word)


def dyslexic_swap_match(a, b):
    """
    See if the two strings are one dyslexic horizontal swap mistake away from correct
    """
    if len(a) != len(b):
        return False
    if a == b:
        return False

    for ii in range(len(a) - 1):
        # using copy here so as not to mutate original string
        a2 = list(copy.deepcopy(a))
        a2[ii] = a[ii + 1]
        a2[ii + 1] = a[ii]
        if ''.join(a2) == b:
            return True

    return False


def dyslexic_flip_match(a, b):
    """
    See if the two strings are one dyslexic vertical flip mistake away from correct
    """
    #     todo: count how many dyslexic flips are needed to make them match rather than True/False
    if len(a) != len(b):
        return False
    if a == b:
        return False

    dyslexic_map = {'9': '6',
                    '6': '9'}

    for ii in range(len(a)):
        # using copy here so as not to mutate original string
        a2 = list(copy.deepcopy(a))
        if a[ii] in dyslexic_map:
            a2[ii] = dyslexic_map[a[ii]]

        if ''.join(a2) == b:
            return True

    return False


def set_match(a, b):
    """
    See if strings a and b contain the same characters but the wrong order
    """
    a2 = ''.join(sorted(list(a)))
    b2 = ''.join(sorted(list(b)))
    return a2 == b2


def round_metrics(w):
    return {k: v if not etl.isdigit(str(v)) else np.around(v, decimals=4) for k, v in w.items()}

# todo: move to the right subplace
def process_vor(time_df, audio_dict):
    start_idx, _ = utils_df.get_exam_start(time_df)

    results = {'vor_lag': {},
               'vor_lags_median': {}}

    eye_L = time_df['Left Eye Position X'].iloc[:start_idx]
    leftEyeXZScore = (eye_L - np.nanmean(eye_L)) / np.nanstd(eye_L)
    eye_R = time_df['Right Eye Position X'].iloc[:start_idx]
    rightEyeXZScore = (eye_R - np.nanmean(eye_R)) / np.nanstd(eye_R)
    head = time_df['Camera Position X'].iloc[:start_idx]
    headPosXZScore = (head - np.nanmean(head)) / np.nanstd(head)
    t = time_df['Total Time'].iloc[:start_idx] - time_df['Total Time'].loc[0]
    ds = np.median(time_df['Delta Time'].iloc[:start_idx])

    # put all signals on the same time scale
    t_i, eye_L = etl.interp_nans(t, eye_L)
    _, eye_R = etl.interp_nans(t, eye_R, t_i)
    _, head = etl.interp_nans(t, head, t_i)

    # time = cumsum(vor.t);

    # for each eye - compute the lag
    results['vor_lag'] = {cur_eye:
                              chunk_2_lag(list(zip(time_df[cur_eye + ' Eye Position X'], headPosX)), ds)
                          for cur_eye in ['Left', 'Right']
                          }

    # compute median lag over a bunch of different sections
    num_parts = 5
    for cur_eye in ['Left', 'Right']:
        chunks = etl.chopn(list(zip(time_df[cur_eye + ' Eye Position X'], headPosX)), num_parts)
        cur_maxlags = [chunk_2_lag(cur_chunk, ds) for cur_chunk in chunks]
        results['vor_lags_median'][cur_eye] = np.median(cur_maxlags)

    # print(results['vor_lag']['Left'], results['vor_lag']['Right'])
    # main_result = np.mean([results['vor_lag']['Left'], results['vor_lag']['Right']])

    results['main_result'] = 0
    results['l_eye_x'] = leftEyeX
    results['l_eye_x_zscore'] = leftEyeXZScore
    results['r_eye_x'] = rightEyeX
    results['r_eye_x_zscore'] = rightEyeXZScore
    results['head_pos_x'] = headPosX
    results['head_pos_x_zscore'] = headPosXZScore
    results['t'] = t

    metrics = {'phase_lag': np.mean([results['vor_lag']['Left'], results['vor_lag']['Right']]),
               'bpm': 5,
               'nystagmus': "No",
               'next_exam_params': None}

    return {'data': results,
            'metrics': metrics,
            'has_error': False
            }


def norm_pupil(pupil, stim_idx):
    z_mean = np.mean([np.nanmedian(np.real(pupil)[stim_idx - 5: stim_idx]),
                      np.nanmedian(np.imag(pupil)[stim_idx - 5: stim_idx])])
    normed_p = pupil - z_mean - 1j * z_mean
    return normed_p


# todo: actually implement process_vorx
def process_vorx(time_df, audio_dict):
    results = {}
    metrics = {'next_exam_params': None}
    return {'data': results,
            'metrics': metrics,
            'has_error': False
            }

