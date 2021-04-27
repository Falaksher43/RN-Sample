from analysis.a_general import *


def process_pupillaryreflex(time_df, audio_dict):
    t, pupil, stim = utils_df.get_t_pupil(time_df)

    stim_idx_tmp = etl.start_and_ends(stim)
    print(stim_idx_tmp)
    if len(stim_idx_tmp) > 0:
        stim_start_end = stim_idx_tmp[0]
        stim_idx = stim_idx_tmp[0][0]
        # stim_idx = etl.start_and_ends(stim)[0][0]
        pupil = norm_pupil(pupil, stim_idx)
        t = t - min(t)
    else:
        logger.warning('Issue analysis @ Line 382')
        stim_start_end = np.nan
        t = np.array([])
        pupil = np.array([])
        stim_idx = np.nan

    if len(pupil) > 0:
        pupil_max_dilation = -np.nanmin(pupil.real)
        pupil_max_constriction = np.nanmedian(pupil[stim_start_end[0]:stim_start_end[0] + 5])
        pupil_range = pupil_max_constriction - pupil_max_dilation
        pupil_dilation_velocity = np.nanmin(np.diff(pupil.real[stim_start_end[0]:stim_start_end[1]]))
    else:
        pupil_max_dilation = np.nan
        pupil_max_constriction = np.nan
        pupil_range = np.nan
        pupil_dilation_velocity = np.nan

    results = {'t': t,
               'pupil': pupil,
               'stim_idx': stim_idx,
               'stim_start_end': stim_start_end
               }

    # todo: update aniscoria here
    metrics = {'pupil_max_dilation': pupil_max_dilation,
               'pupil_max_constric': np.around(np.abs(pupil_max_constriction), decimals=2),
               'pupil_range': np.abs(pupil_range),
               'pupil_dilation_velocity': np.around(pupil_dilation_velocity, decimals=2),
               'aniscoria': "No",
               'next_exam_params': None
               }

    return {'data': results,
            'metrics': metrics,
            'has_error': False
            }

def process_prosaccade(time_df, audio_dict):
    def gather_normal_prosaccades():
        """Using the file structure get all the normal data"""
        # todo: reformat to get these from the database
        # our concussion patient
        cidd = '09539aac-01f6-4332-ad94-9063b3160f48'

        data_folders = utils_df.get_data_folders('data/Good Data')
        results = utils_df.load_results(data_folders)
        #     convert the results into a data frame
        # convert the results dictionary into a dataframe
        cols = ['reaction_times', 'stim_starts', 'stim_ends', 'eye_starts', 'eye_ends', 'eye_fixs', 'max_speeds']
        all_results = []
        for k, v in results.items():
            df = pd.DataFrame.from_dict({k: v['prosaccade'][k] for k in cols})
            df['id'] = v['id']
            df['drinks'] = v['drinks']
            all_results.append(df)
        # get everything in one DF
        sdf = pd.concat(all_results)
        sdf['concussion'] = sdf['drinks'].apply(lambda w: 'concussed' if w == 'concussion' else 'normal')
        sdf['distance'] = np.round(10 * np.abs(sdf['stim_ends'] - sdf['stim_starts'])) / 10.0

        # quantify the errors
        sdf['error_vec'] = sdf['eye_fixs'] - sdf['eye_ends']
        sdf['error_mag'] = np.abs(sdf['error_vec'])

        # error in the direction of the saccade
        # dot product of stim direction vector - and error vec.
        sdf['stim_vec'] = sdf['stim_ends'] - sdf['stim_starts']
        stim_vec = sdf['stim_vec']
        sdf['overshoot'] = etl.complex_dot(sdf['error_vec'].values, sdf['stim_vec'].values)

        # calibrate people's positions
        sdf['stim_ends_r'] = np.around(sdf['stim_ends'], decimals=3)
        sdf['stim_starts_r'] = np.around(sdf['stim_starts'], decimals=3)
        stim_pos = np.unique(sdf['stim_ends_r'])
        calb_dict = dict()
        for pos in stim_pos:
            #     2D median
            z = sdf['eye_ends'][sdf['stim_ends_r'] == pos].values
            x, y = etl.geometric_median(np.array([z.real, z.imag]).T)
            calb_dict[pos] = x + 1j * y

        sdf['stim_ends_c'] = [calb_dict[pos] for pos in sdf['stim_ends_r']]
        sdf['stim_starts_c'] = [calb_dict[pos] for pos in sdf['stim_starts_r']]
        sdf['stim_vec_c'] = sdf['stim_ends_c'] - sdf['stim_starts_c']
        sdf['error_c'] = sdf['stim_ends_c'] - sdf['eye_ends']
        sdf['overshoot_c'] = etl.complex_dot(sdf['error_c'].values, sdf['stim_ends_c'].values)

        # for cnt, tmp in enumerate([.5, 1]):
        # I = sdf['distance'] == tmp
        norm_stims = stim_vec
        norm_eye_ends = sdf['error_c'] * np.exp(-1j * np.angle(stim_vec)) / np.abs(stim_vec)
        norm_reaction_times = sdf['reaction_times']
        norm_distances = sdf['distance']

        df = pd.DataFrame.from_dict({'norm_stims': norm_stims,
                                     'norm_eye_ends': norm_eye_ends,
                                     'norm_reaction_times': norm_reaction_times,
                                     'norm_distances': norm_distances})

        return calb_dict, df

    t, eye, stim = utils_df.get_t_z_stim(time_df)
    saccades_df = saccades.cget_saccades(eye, t)

    # filter out saccades that are too small
    saccades_df = saccades_df[saccades_df['distances'] > .1]

    # find if the stimulus moved in the or y direction
    stim_moves = np.abs(np.diff(stim))
    moves = np.nan_to_num(stim_moves).astype(bool)

    # find the moments when the stimulus changes
    move_idxs = np.arange(len(moves))[moves]

    # measure many things
    # the stimulus and eye positions at the end of the first saccade

    # complex outputs
    stim_starts = []
    stim_ends = []
    eye_starts = []
    eye_ends = []
    eye_fixs = []  # the fixation just before the next stim move

    # timeseries outputs
    eye_timeseries = []
    t_timeseries = []

    # scalar outputs
    reaction_times = []
    max_speeds = []
    saccade_count_per_path = []

    m = 2  # the added bits from the end of the V to when the saccade really ends
    n = 10
    # saccade starts and ends, units idx
    ends = np.array([w[1] for w in saccades_df['saccades']])
    starts = np.array([w[0] for w in saccades_df['saccades']])
    for ii, idx in enumerate(move_idxs):

        # collect the positions and movements of the dots
        stim_start = stim[idx]
        stim_end = stim[idx + 1]
        stim_starts.append(stim_start)
        stim_ends.append(stim_end)

        # find the saccade that made the movement!
        end_idx = ends[ends > idx]
        start_idx = starts[starts > idx]
        if len(end_idx) > 0 and len(start_idx) > 0:
            end_idx = end_idx[0]
            start_idx = start_idx[0]
        else:  # have not found a saccade meeting our criteria
            continue

        # count how many saccades per movement/path
        # set current movement stimulus
        count = 0
        lower_bound = move_idxs[ii]

        # check to make sure there's no index out of bounds error
        if (ii != len(move_idxs) - 1):
            # set next movement stim
            upper_bound = move_idxs[ii + 1]
            # identify saccades that fall between lower and upper bound
            start_set = starts[starts > lower_bound]
            start_set = start_set[start_set <= upper_bound]
            end_set = ends[ends > lower_bound]
            end_set = end_set[end_set <= upper_bound]
        else:
            upper_bound = move_idxs[ii]
            start_set = starts[starts > lower_bound]
            end_set = ends[ends > lower_bound]

        # count how many saccades are within range
        if len(start_set) > 0 and len(end_set) > 0:
            count = len(start_set)

        saccade_count_per_path.append(count)

        # saccade begin - stim move
        reaction_times.append(t[start_idx] - t[idx])

        eye_start = eye[idx]
        eye_end = eye[end_idx + m]
        eye_starts.append(eye_start)
        eye_ends.append(eye_end)

        # add in the last eye position before the stim moved
        eye_fixs.append(eye[idx - 1])

        # gather the first second of t and z after saccade starts
        time_to_track = 1.5  # seconds after saccade to return
        # if you are on the last thing
        if idx == move_idxs[-1]:
            max_t = t[start_idx] + time_to_track
        else:
            # print(t[move_idxs[ii + 1]])
            # print(t[start_idx] + time_to_track)

            max_t = np.min([t[move_idxs[ii + 1]], t[start_idx] + time_to_track])

        # saccade started, and at least this much later
        I = np.logical_and(t > t[start_idx], t < max_t)
        # eye_timeseries.append([t[I], eye[I]])
        eye_timeseries.append(eye[I])
        t_timeseries.append(t[I])

        dists = np.abs(np.diff(eye[start_idx:end_idx]))
        if len(dists) > 0:
            max_speeds.append(np.max(dists))
        else:
            max_speeds.append(0)

    # remove the first element, so the fixation is for the one you saccaded to.
    eye_fixs = eye_fixs[1:] + [np.nan]

    # build results dictionary for export
    results = dict()
    results['number of saccades'] = len(saccades_df)
    results['saccades per movement'] = len(saccades_df) / (len(stim_starts) + etl.eps)
    results['median reaction time'] = np.nanmedian(reaction_times)
    results['main_result'] = results['median reaction time']
    results['duration'] = t[-1] - t[0]
    results['reaction_times'] = list(reaction_times)
    results['eye_timeseries'] = np.array(eye_timeseries)
    results['t_timeseries'] = np.array(t_timeseries)
    results['stim_starts'] = np.nan_to_num(np.array(stim_starts), copy=False, nan=0)  # TODO: is this ok?
    results['stim_ends'] = np.nan_to_num(np.array(stim_ends), copy=False, nan=0)  # TODO: is this ok?
    results['eye_starts'] = np.array(eye_starts)
    results['eye_ends'] = np.array(eye_ends)
    results['eye_fixs'] = np.nan_to_num(np.array(eye_fixs), copy=False, nan=0)  # TODO: is this ok?
    results['reaction_times'] = list(reaction_times)
    results['max_speeds'] = list(max_speeds)
    results['saccade_count_per_path'] = list(saccade_count_per_path)

    # record the index
    results['sacc_startidx'] = starts
    results['sacc_endidx'] = ends

    # build metrics dictionary for summary report usage
    metrics = dict()
    metrics['number_of_saccades'] = len(saccades_df)
    metrics['saccades_per_movement'] = np.around(len(saccades_df) / (len(stim_starts) + etl.eps), decimals=2)
    metrics['median_reaction_time'] = np.around(np.nanmedian(reaction_times), decimals=2)
    metrics['duration'] = np.around(t[-1] - t[0], decimals=2)
    metrics['abnormal_path_proportion'] = np.around(
        sum(map(lambda i: i > 2, results['saccade_count_per_path'])) / results['number of saccades'],
        decimals=2)
    metrics['next_exam_params'] = None

    return {'data': results,
            'metrics': metrics,
            'has_error': False,
            }
    # return pd.Series(results)
    # cols = ['reaction_times', 'stim_starts', 'stim_ends', 'eye_starts', 'eye_ends', 'eye_fixs', 'max_speeds']
    # df = pd.DataFrame.from_dict({k: results[k] for k in cols})

    # return df


def p_results_to_sdf(result):
    """convert the results dictionary into a dataframe"""

    cols = ['reaction_times', 'stim_starts', 'stim_ends', 'eye_starts', 'eye_ends', 'eye_fixs', 'max_speeds',
            'eye_timeseries', 't_timeseries', 'saccade_count_per_path']
    sdf = pd.DataFrame.from_dict({k: result[k] for k in cols})
    sdf['stim_vec'] = np.around(sdf['stim_ends'] - sdf['stim_starts'], decimals=3)
    sdf['distance'] = np.round(10 * np.abs(sdf['stim_ends'] - sdf['stim_starts'])) / 10.0

    # remove the last stim + movement because people behave differently when the task is complete
    sdf = sdf.iloc[0:-1]
    return sdf



def process_smoothpursuit(time_df, audio_dict):
    """
    We use a cross correlation to measure the lag. Therefore we need the data to be
     - without any nans
     - with a constant sampling rate
    And so (after transform t_z_stim which just adjust calibration, and sets start time to zero)
    we apply interp_nans which will interpolate over and missing
    data points that are nans and simultaneously standardize the sampling rate over a new time 't_i'
    After setting t_i with the combined eye data - we interpolate every other signnal over the same
    interpolated time.

    To catch in case folks lose lock part-way through the exam we chunk the data up into 5 parts
    (num_parts) and calculte the lag for each section.

    """

    t, z, stim = utils_df.get_t_z_stim(time_df, 'Combine')
    t, z, stim = transform_t_z_stim(t, z, stim)
    # interpolate over any nans and also make the sampling rate consistent
    t_i, x = etl.interp_nans(t, np.real(z))
    _, stim_i = etl.interp_nans(t, np.real(stim), t_i=t_i)

    _, zL, stim = utils_df.get_t_z_stim(time_df, 'Left')
    _, zL, stim = transform_t_z_stim(t, zL, stim)
    _, L = etl.interp_nans(t, np.real(zL), t_i=t_i)

    _, zR, stim = utils_df.get_t_z_stim(time_df, 'Right')
    _, zR, stim = transform_t_z_stim(t, zR, stim)
    _, R = etl.interp_nans(t, np.real(zR), t_i=t_i)

    t = t_i
    stim = stim_i

    # err_position = error_position(z, stim)

    results = {'data': {
        't': t,
        'x': x,
        'L': L,
        'R': R,
        'stim': stim
    },
    }
    smooth = dict()
    smooth['leftEyeX'] = L  # time_df['Left Eye Position X']
    smooth['rightEyeX'] = R  # time_df['Right Eye Position X']

    smooth['stimulusX'] = stim  # utils_df.robust_get_stimulus(time_df, xy='X', num='1')

    # smooth['t'] = smooth_data['Dot1 Position X']
    ds = np.median(time_df['Delta Time'])
    results['ds'] = ds
    # smooth['time'] = np.cumsum(smooth['t'])

    # smooth the movement
    # todo: smooth using x and y together
    smooth['leftEyeX'] = etl.nan_smooth(smooth['leftEyeX'], n=15)
    smooth['rightEyeX'] = etl.nan_smooth(smooth['rightEyeX'], n=15)

    # for each eye - compute the lag
    results['smooth_lag'] = {cur_eye:
                                 chunk_2_lag(list(zip(smooth[cur_eye + 'EyeX'], smooth['stimulusX'])), ds)
                             for cur_eye in ['left', 'right']
                             }

    results['smoothLagsMedian'] = dict()
    results['data']['lags_all'] = dict()
    # compute median lag over a bunch of different sections
    num_parts = 5
    for cur_eye in ['left', 'right']:
        chunks = etl.chopn(list(zip(smooth[cur_eye + 'EyeX'], smooth['stimulusX'])), num_parts)
        cur_maxlags = [chunk_2_lag(cur_chunk, ds) for cur_chunk in chunks]
        results['smoothLagsMedian'][cur_eye] = np.nanmedian(cur_maxlags)
        results['data']['lags_all'][cur_eye] = cur_maxlags

        # results['main_result'] = np.median(cur_maxlags)
    results['metrics'] = {
        'median_lag_left': results['smoothLagsMedian']['left'],
        'median_lag_right': results['smoothLagsMedian']['right'],
        'next_exam_params': None
    }

    results['has_error'] = False

    return results

