# React Track,
# Package for analyzing x/y of eye movements and extracting features (saccades, fixations etc)


from matviz.matviz import etl
import numpy as np
import pandas as pd
import warnings

def get_features(z, t):
    """todo: export fancier features"""

    features = dict()


    # results = {'folder': cur_folder}

    # process sacaddes to get reaction times
    # features = {**features, **process_saccades(x, y, t)}

    # process smooth pursuit
    # features = {**features, **process_smooth_pursuit(cur_folder)}







def cget_saccades(z, t, thresh_velocity = 5, thresh_distance=.025):
# def cget_saccades(z, t, saccade_thresh = .02):
    """
    # get saccades passing in a complex array
    # def cget_saccades(z, t, saccade_thresh = .03):
    :param z: x + iy eye directions
    :param t: time
    :param thresh_velocity: the threshold between saccades and fixations
    :param thresh_distance: the distance a saccade needs to traverse to not be removed

    :return: a dataframe where each row is a saccade
    """

    # check that the input is correct:
    if type(t) not in [np.ndarray, list]:
        raise Exception("t must be ndarray, you passed: " + str(type(t)))

    if not (hasattr(z, 'dtype') and z.dtype == 'complex128'):
        raise Exception("z must be ndarray and complex, you passes: "  + str(type(z)))

    if len(z) != len(t):
        raise Exception("Length of 'z' and 't' must be equal, you passed: len(z)=" + str(len(z)) + " t=" + str(len(t)))

    # v is speed
    v = np.abs(np.diff(z)) / np.diff(t)
    with warnings.catch_warnings():
        # ignore a warning for doing a '>' on a NaN, they return false and are not saccades
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        saccades = etl.start_and_ends(v > thresh_velocity)

    # enforce there be a least 2 sample length (in case it starts with a 0 length saccade)
    saccades = [w for w in saccades if w[1] - w[0] > 0]

    saccade_times = [[t[w[0]], t[w[1]]] for w in saccades]
    saccade_starts = [w[0] for w in saccade_times]

    saccade_durations = [ w[1] - w[0] for w in saccade_times]
    saccade_vectors = [z[endd] - z[start] for start, endd, in saccades]

    speeds_max = [np.max(v[start:endd]) for start, endd, in saccades]
    distances = [np.abs(w) for w in saccade_vectors]
    angles = [np.angle(w) for w in saccade_vectors]

    results_dict =  {'saccades': saccades,
            'starts': saccade_starts,
            'durations': saccade_durations,
            'vectors': saccade_vectors,
            'distances': distances,
            'angles': angles,
            'speeds_max': speeds_max}

    saccades_df = pd.DataFrame.from_dict(results_dict).dropna()

    # remove saccades that are too small
    saccades_df = saccades_df[saccades_df['distances'] > thresh_distance]

    return saccades_df

# get saccades passing in x and y instead of z
def get_saccades(x, y, t, **kwargs):
    return cget_saccades(x + 1j * y, t, **kwargs)





