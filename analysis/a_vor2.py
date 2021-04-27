from analysis.a_general import *



def process_convergence(time_df, audio_dict, type='welch', compute_spectrogram=True):
    def transform_convergence_stim(stimz):
        """
        Convert stimulus z values into expected 'x' changes for an eye
        with distance from eye to center of R, and distance of eyes to origin of A
        6 is an arbitrary scale factor, though the other factors are also a little arbitrary
        See more documentation here https://www.notion.so/reactneuro/Visualize-eye-convergence-f3d201016c504e79bd161154dfc825d4
        :param stim:
        :return:
        """
        R = 5
        A = 1
        stim_transformed = -np.arctan((R - 6 * stimz) / A) / 10
        return stim_transformed

    ds = np.nanmedian(time_df['Delta Time'])

    t, left, right = utils_df.get_t_lr(time_df)

    ti, x_L = etl.interp_nans(t, left.real)
    ti, x_R = etl.interp_nans(t, right.real)
    ti, y_L = etl.interp_nans(t, left.imag)
    ti, y_R = etl.interp_nans(t, right.imag)

    signals = [x_L, x_R, y_L, y_R]
    if type == 'periodogram':
        power = [signal.periodogram(s, 1 / ds, window='hann') for s in signals]
    elif type == 'welch':
        power = [signal.welch(s, 1 / ds, window='hann') for s in signals]

    start_idx, _ = utils_df.get_exam_start(time_df)
    _, stim = etl.interp_nans(time_df['Total Time'], utils_df.robust_get_stimulus(time_df, xy='Z'), ti)
    # convert stim to how the eyes_x will move
    stim_transformed = transform_convergence_stim(stim)
    # calculate the phase lag for left and right eyes
    I = np.logical_and(etl.rolling_diff(stim, n=1) != 0, stim > 0)
    # todo: calculate the stimulus frequncy, and set the max_lag_allowed to be half of that
    lag_L, corr_L = etl.max_lag(np.real(x_L[I]), stim_transformed[I], ds, max_lag_allowed=5)
    lag_R, corr_R = etl.max_lag(np.real(-x_R[I]), stim_transformed[I], ds, max_lag_allowed=5)

    lag_LR, corr_LR = etl.max_lag(np.real(x_L[I]), -    np.real(x_R[I]), ds)

    results = dict()
    results['x_L'] = x_L[I]
    results['x_R'] = x_R[I]
    results['y_L'] = y_L[I]
    results['y_R'] = y_R[I]
    results['t'] = ti[I]
    results['stim'] = stim[I]
    results['stim_transformed'] = stim_transformed[I]
    results['lag_L'] = lag_L
    results['lag_R'] = lag_R
    results['lag_LR'] = lag_LR
    results['corr_L'] = corr_L
    results['corr_R'] = corr_R
    results['corr_LR'] = corr_LR

    results['freq'] = power[0][0]
    results['x_L_power'] = power[0][1]
    results['x_R_power'] = power[1][1]
    results['y_L_power'] = power[2][1]
    results['y_R_power'] = power[3][1]

    results['ds'] = ds

    if compute_spectrogram == True:
        power_spectrum = [signal.spectrogram(s, 1 / ds, nperseg=100, noverlap=90) for s in signals]

        results['f_spectrogram'] = power_spectrum[0][0]
        results['t_spectrogram'] = power_spectrum[0][1]
        results['x_L_spectrogram'] = power_spectrum[0][2]
        results['x_R_spectrogram'] = power_spectrum[1][2]
        results['y_L_spectrogram'] = power_spectrum[2][2]
        results['y_R_spectrogram'] = power_spectrum[3][2]

    metrics = {'stimulus_phase_lag': np.abs(np.mean([lag_L, lag_R])),
               'stimulus_correlation': np.mean([corr_L, corr_R]),
               'LR_phase_lag': lag_LR,
               'LR_correlation': corr_LR,
               'next_exam_params': None}

    return {'data': results,
            'metrics': metrics,
            'has_error': False}


def process_smoothpursuit2d(time_df, audio_dict):
    t, z, stim = utils_df.get_t_z_stim(time_df)
    t, z, stim = transform_t_z_stim(t, z, stim)

    err_position = error_position(z, stim)

    results = {'data': {
        't': t,
        'z': z,
        'stim': stim
    },
        'metrics': {
            'error_magnitude': np.nanmedian(np.abs(err_position)),
            'error_angular': np.nanmedian(np.rad2deg(err_position.imag)),
            'error_radial': np.nanmedian(err_position.real),
            'next_exam_params': None
        },
        'has_error': False
    }

    return results


def process_selfpacedsaccade(time_df, audio_dict):
    """
    Process self guided saccade digital exam
    :param time_df:
    :return results: dictionary of results
    """
    # get stimlus and eye data starting at beginning of digital exam
    t, eye, stims = utils_df.get_t_z_stims(time_df)

    # get saccades
    saccades_df = saccades.cget_saccades(eye, t)

    # separate normal and microsaccades
    normal_saccades_df = saccades_df[saccades_df['distances'] > 0.6]
    microsaccades = saccades_df[saccades_df['distances'] <= 0.6]

    # calculate long saccade velocities
    saccades_velocity = (saccades_df.distances / saccades_df.durations).values
    median_sacc_vel = np.nanmedian(saccades_velocity)

    # calculate saccade accuracy
    saccades_accuracy = 1

    results = {'t': t,
               'eye': eye,
               'stim': stims,
               'num_saccades': len(saccades_df),
               'saccades': saccades_df,
               'normal_saccades': normal_saccades_df,
               'microsaccades': microsaccades,
               'saccades_velocity': saccades_velocity}

    metrics = {'sacc_per_sec': np.around(len(saccades_df) / (t[-1] - t[0]), decimals=2),
               'median_vel': np.around(median_sacc_vel, decimals=2),
               'vel_acc': np.around(median_sacc_vel / saccades_accuracy, decimals=2),
               'accuracy': saccades_accuracy,
               'next_exam_params': None}

    return {'data': results,
            'metrics': metrics,
            'has_error': False
            }

