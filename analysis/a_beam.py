
from analysis.a_general import *
from collections import defaultdict
import pandas as pd
import json



def process_beam(time_df, audio_dict):

    stim_df = get_stim_df(time_df)
    results_df = get_results_df(stim_df)

    results_json = get_results_dict(results_df)

    results = {
                'results_df': results_df,
                'stim_df': stim_df,
                'results_json': results_json
               }


    return results



def get_results_dict(results_df):


    # converty numpy arrays into json serializable items
    itemize = lambda x: [w.item() for w in x]

    cuttoff_n = 22

    sub_sample_factor = 5

    def saccades_for_charting(results_df):
        C = get_colormap()
        eye_movements = []
        for ii in range(len(results_df)):      #                      round to 1 decimal, subsample, then take first _n samples
            x = itemize(np.around(np.real(results_df.iloc[ii]['eye_z']), decimals=1))[::sub_sample_factor][:cuttoff_n]
            y = itemize(np.around(np.imag(results_df.iloc[ii]['eye_z']), decimals=1))[::sub_sample_factor][:cuttoff_n]
            c = C[ii]
            #     plot(x, y, color=C[ii])

            eye_movements.append({'x': x, 'y': y, 'color': c})
        return eye_movements


    def saccades_for_charting_tx(results_df):
        # fancy version, looks awesome but currently unused
        factor = 15
        C = get_colormap()
        eye_movements = []
        for ii in range(len(results_df)):
            x = itemize(np.around(results_df.iloc[ii]['time'], decimals=3))[::sub_sample_factor]
            y = itemize(np.around(results_df.iloc[ii]['eye_x'] + factor * ii, decimals=3))[::sub_sample_factor]
            c = itemize((np.array(C[ii]) * 255).astype(int))
            #     plot(x, y, color=C[ii])

            eye_movements.append({'x': x, 'y': y, 'color': c})
        return eye_movements


    def get_group_results(results_df):
        mapper = {'D': 'Directional',
                  'M': 'Misdirectional',
                  'U': 'Uncued',
                  'N': 'No-Go'}

        results_df['group'] = results_df['trial_type'].apply(lambda w: mapper[w[0]])

        saccadic_reaction_times = defaultdict(dict)
        manual_reaction_times = defaultdict(dict)

        for k, group in results_df.groupby('group'):

            if k == 'No-Go':
                nogo_errors = {
                    'Saccadic': {'value': np.sum(~group['eye_correct']).item()},
                    'Manual': {'value': np.sum(~group['thumb_correct']).item()}
                }
            else:
                saccadic_reaction_times[k]['mean'] = np.around(np.nanmean(group['response_time_eye']),
                                                               decimals=4).item()
                saccadic_reaction_times[k]['std'] = np.around(np.nanstd(group['response_time_eye']), decimals=4).item()

                manual_reaction_times[k]['mean'] = np.around(np.nanmean(group['response_time_thumb']),
                                                             decimals=4).item()
                manual_reaction_times[k]['std'] = np.around(np.nanstd(group['response_time_thumb']), decimals=4).item()

        return {"Saccadic RT": saccadic_reaction_times,
                "Manual RT": manual_reaction_times,
                "No-Go Errors": nogo_errors}

    eye_movements = saccades_for_charting(results_df)
    group_results = get_group_results(results_df)

    combined_data = {
                        'eye movements': eye_movements,
                        'bar grahs': group_results
                    }

    return combined_data


def get_stim_df(time_df):

    starts_ends = start_and_ends(time_df['Trial Type'].astype('category').cat.codes.values)

    stim_df = pd.DataFrame()
    for start, endd in starts_ends:
        sub_df = time_df.iloc[start + 1:endd]

        response_begin_idx = np.argmax(sub_df['Expected Input'].values)

        trail_type = sub_df.iloc[0]['Trial Type']
        expected_input = sub_df.iloc[response_begin_idx]['Expected Input']
        expected_focus_z = sub_df.iloc[response_begin_idx]['Expected Focus X'] + 1j * sub_df.iloc[0]['Expected Focus Y']
        start_time = sub_df.iloc[response_begin_idx]['Total Time']

        # remove crazy values of X
        clean_x = lambda w: w if -5 < w < 20 else np.nan

        sub_dict = {
            'Trial Type': trail_type,
            'Expected Thumb Text': expected_input,
            'Expected Thumb Z': thumb_direction_to_z(expected_input),
            'Expected Focus Z': expected_focus_z,
            'cue idx': int(response_begin_idx),
            'time': sub_df['Total Time'].values - start_time,
            'eye z': sub_df['Gaze Intersection X'].apply(clean_x).values + 1j * sub_df['Gaze Intersection Y'].values,
            'thumb z': sub_df['Right Thumbstick X'].values + 1j * sub_df['Right Thumbstick Y'].values,
        }

        stim_df = stim_df.append(sub_dict, ignore_index=True)

    stim_df['cue idx'] = stim_df['cue idx'].astype(int)

    return stim_df

def thumb_direction_to_z(thumb_text):

    if thumb_text == 'LEFT':
        thumb_z = -1
    elif thumb_text == 'RIGHT':
        thumb_z = 1
    if thumb_text == 'UP':
        thumb_z = 1j
    elif thumb_text == 'DOWN':
        thumb_z = -1j
    elif thumb_text == 'NONE':
        thumb_z = 0

    return thumb_z

def get_results_df(stim_df):

    def evaluate_individual_stim(stim):
        """
        Measure accuracy

        :param stim:
        :return:
        """

        THRESHHOLD_THUMB = 0.5
        THRESHHOLD_EYE = 5

        def get_response_time(time, cue_idx, z, focus, thresh):
            # find the times when the eye is inside the target, and *after the cue
            inside_target = np.abs(z[cue_idx:] - focus) < thresh
            response_time = time[cue_idx:][np.argmax(inside_target)]
            # remove any where we didn't find any at all (so argmax returns 0)
            response_time = np.nan if response_time == 0 else response_time
            return response_time

        cur_result = dict()

        # pop in the raw movement data
        cur_result['eye_x'] = np.real(stim['eye z'])
        cur_result['eye_z'] = stim['eye z']
        cur_result['thumb_x'] = np.real(stim['thumb z'])
        cur_result['time'] = np.real(stim['time'])

        cur_result['trial_type'] = stim['Trial Type']
        cur_result['expected_text']= stim['Expected Thumb Text']


        #     thumbs at some point got to the right spot
        cur_result['thumb_direction_correct'] = np.min(np.abs(stim['thumb z'] - stim['Expected Thumb Z'])) < THRESHHOLD_THUMB
        #      thumbs are close to the wrong spot at least some time               -1 * expected value, flips on appropriate axis
        cur_result['thumb_direction_inccorrect'] = np.min(np.abs(stim['thumb z'] + stim['Expected Thumb Z'])) < THRESHHOLD_THUMB
        #
        cur_result['thumb_correct'] = cur_result['thumb_direction_correct'] and ~cur_result['thumb_direction_inccorrect']
        cur_result['thumb_correction'] = cur_result['thumb_direction_correct'] and cur_result['thumb_direction_inccorrect']


        #     eyes are in the right spot at least at some point
        cur_result['eye_direction_correct'] = np.min(np.abs(stim['eye z'] - stim['Expected Focus Z'])) < THRESHHOLD_EYE
        #      eyes are close to the wrong spot at least some time               -1 * expected value, flips on x axis
        cur_result['eye_direction_inccorrect'] = np.min(np.abs(stim['eye z'] + stim['Expected Focus Z'])) < THRESHHOLD_EYE
        #                                   you ended in the right place and you didn't go to the wrong place
        cur_result['eye_correct'] = cur_result['eye_direction_correct'] and ~cur_result['eye_direction_inccorrect']
        cur_result['eye_correction'] = cur_result['eye_direction_correct'] and cur_result['eye_direction_inccorrect']


        # calculate the response times
        cue_idx = stim['cue idx']
        cur_result['response_time_eye'] = get_response_time(stim['time'], cue_idx, stim['eye z'], stim['Expected Focus Z'], THRESHHOLD_EYE)
        cur_result['response_time_thumb'] = get_response_time(stim['time'], cue_idx, stim['thumb z'], stim['Expected Thumb Z'], THRESHHOLD_THUMB)


        return cur_result

    results_df = pd.DataFrame(stim_df.apply(evaluate_individual_stim, axis=1).to_list())

    return results_df





def start_and_ends(logical_array):
    """
     Return the starts and end times for when the logical
     array True
    :param logical_array:
    :return:
    list of (start,end) tuples of the indexes

    Note: if the array starts with a [True, False], you completely
          miss it because it technically *ended at that point
          and started before the logical array began
          If the array starts with [True,True, False]
          then you get [(0,1),...]
    #

    """

    # Padd the array with Falses to get the ends
    padded_array = np.concatenate(([False], logical_array, [False]))

    #
    idxs = np.array(range(len(padded_array) - 1))
    differences = np.diff([np.int(w) for w in padded_array])
    starts = idxs[differences > 0]
    ends   = idxs[differences < 0]

    # we added an element, now we take it away
    starts_shift = np.maximum(starts - 1, 0)
    # easier than doing a check if its empty
    ends_shift = np.maximum(ends - 1, 0)

    return list(zip(starts_shift, ends_shift))



def rbg2hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2]) * 255)

def get_colormap():
    tmp_array = ([[0.272594, 0.025563, 0.353093],
           [0.277018, 0.050344, 0.375715],
           [0.280894, 0.078907, 0.402329],
           [0.282656, 0.100196, 0.42216],
           [0.283229, 0.120777, 0.440584],
           [0.28229, 0.145912, 0.46151],
           [0.280255, 0.165693, 0.476498],
           [0.277134, 0.185228, 0.489898],
           [0.271828, 0.209303, 0.504434],
           [0.26658, 0.228262, 0.514349],
           [0.260571, 0.246922, 0.522828],
           [0.252194, 0.269783, 0.531579],
           [0.244972, 0.287675, 0.53726],
           [0.237441, 0.305202, 0.541921],
           [0.227802, 0.326594, 0.546532],
           [0.220057, 0.343307, 0.549413],
           [0.212395, 0.359683, 0.55171],
           [0.203063, 0.379716, 0.553925],
           [0.19586, 0.395433, 0.555276],
           [0.188923, 0.41091, 0.556326],
           [0.180629, 0.429975, 0.557282],
           [0.174274, 0.445044, 0.557792],
           [0.168126, 0.459988, 0.558082],
           [0.160665, 0.47854, 0.558115],
           [0.154815, 0.493313, 0.55784],
           [0.149039, 0.508051, 0.55725],
           [0.141935, 0.526453, 0.555991],
           [0.136408, 0.541173, 0.554483],
           [0.131172, 0.555899, 0.552459],
           [0.125394, 0.574318, 0.549086],
           [0.121831, 0.589055, 0.545623],
           [0.119738, 0.603785, 0.5414],
           [0.120081, 0.622161, 0.534946],
           [0.123444, 0.636809, 0.528763],
           [0.130067, 0.651384, 0.521608],
           [0.143303, 0.669459, 0.511215],
           [0.157851, 0.683765, 0.501686],
           [0.175707, 0.6979, 0.491033],
           [0.202219, 0.715272, 0.476084],
           [0.226397, 0.728888, 0.462789],
           [0.252899, 0.742211, 0.448284],
           [0.288921, 0.758394, 0.428426],
           [0.319809, 0.770914, 0.411152],
           [0.35236, 0.783011, 0.392636],
           [0.395174, 0.797475, 0.367757],
           [0.430983, 0.808473, 0.346476],
           [0.468053, 0.818921, 0.323998],
           [0.515992, 0.831158, 0.294279],
           [0.555484, 0.840254, 0.269281],
           [0.595839, 0.848717, 0.243329],
           [0.647257, 0.8584, 0.209861],
           [0.688944, 0.865448, 0.182725],
           [0.730889, 0.871916, 0.156029],
           [0.783315, 0.879285, 0.125405],
           [0.82494, 0.88472, 0.106217],
           [0.866013, 0.889868, 0.095953],
           [0.916242, 0.896091, 0.100717],
           [0.9553, 0.901065, 0.118128]])

    return [rbg2hex(w) for w in tmp_array]
