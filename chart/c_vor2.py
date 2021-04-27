from chart.c_general import *


def plot_smoothpursuit2d(processed_df, visit_exam_id=None):
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    result = current_df.iloc[0]['processed']

    controls_exist = control_df is not None
    if controls_exist:
        t_control = np.concatenate([w['data']['t'] for w in control_df['processed'].values])
        z_control = np.concatenate([w['data']['z'] for w in control_df['processed'].values])

    t = result['data']['t']
    z = result['data']['z']
    stim = result['data']['stim']

    subplotter(2, 2, 0)
    plot_eye_speed(z, c=linspecer(1)[0])
    cplot(stim, 'k--', lw=2)
    legend(['eye', 'stim'], loc='best')
    axis('off')
    subplotter(2, 2, 1)
    rad_ylim = 4
    if controls_exist:
        nhist_error_radial(t_control, z_control, rad_ylim)
    plot_error_radial(t, z, stim, rad_ylim)

    subplotter(2, 2, [2, 3])
    if controls_exist:
        nhist_error_angular(t, z, stim)
    plot_error_angular(t, z, stim)
    ylim([-200, 200])
    nicefy(fsize=10)
    # subplotter(3, 2, [4, 5])
    # nhist_error_angular(t, z, stim)
    # plot_error_angular(t, z, stim)
    # nicefy(fsize=10)
    # return


def plot_convergence(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    processed_data = current_df.iloc[0]['processed']['data']
    # run analysis on current exam
    # processed_data = analysis.process_convergence(current_df, type='welch', compute_spectrogram=True)

    # run analyses on population, if present
    if control_df is not None:
        # population_results = pd.DataFrame([analysis.process_convergence(series[1].timeseries, type='welch') for series in control_df.iterrows()])
        # control_pop_results_df = pd.concat((control_df, population_results), axis=1, sort='False')
        control_pop_results_df = control_df['processed'].apply(lambda w: w['data'])
    else:
        control_pop_results_df = None

    # plot visualization for convergence breaks, left and right x eye position
    fig_sizer(6, 16)
    C = linspecer(2)

    # setup figure
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=4, figure=fig, width_ratios=[1.15, 1, 1.15])

    # plot variables
    freq_range = (15, 30)
    t = processed_data['t']
    f = processed_data['freq']
    stim = processed_data['stim_transformed']
    x_L = processed_data['x_L']
    x_R = processed_data['x_R']
    x_L_power = processed_data['x_L_power']
    x_R_power = processed_data['x_R_power']
    t_spec = processed_data['t_spectrogram']
    f_spec = processed_data['f_spectrogram']
    Sxx_L_spec = processed_data['x_L_spectrogram']
    Sxx_R_spec = processed_data['x_R_spectrogram']

    # plot eye position over time
    ax1 = fig.add_subplot(spec[:, 0])
    plot(t, x_L, '.', label="_nolegend_", c=np.array(C[0]) / 2 + .5)
    plot(t, -x_R, '.', label="_nolegend_", c=np.array(C[1]) / 2 + .5)
    # plot the stimulus with arbitrary shift downwards to match eye charts
    plot(t, stim - 0.2, 'k--', lw=2)
    plot(t, nan_smooth(x_L, 50), c=C[0], lw=2)
    plot(t, -nan_smooth(x_R, 50), c=C[1], lw=2)
    xlabel('Time (s)')
    ylabel('Eye position')
    legend(['Stimulus', 'Left eye', 'Right eye'])
    ylim([-.5, 1])

    # plot power spectral densities for current and population
    ax2 = fig.add_subplot(spec[0:2, 1])
    ax3 = fig.add_subplot(spec[2:3, 1])
    ax4 = fig.add_subplot(spec[3:4, 1])

    # need to have control results to plot the below
    if control_pop_results_df is not None:
        for ii, row in control_pop_results_df.iteritems():
            if len(row['freq']) == len(row['x_R_power']) == len(row['x_L_power']):
                ax2.plot(row['freq'], np.log10(row['x_R_power']), 'k', alpha=.05)
                ax2.plot(row['freq'], np.log10(row['x_L_power']), 'k', alpha=.05)

        freq_indices = np.where((control_pop_results_df.iloc[0]['freq'] > freq_range[0]) & (
                    control_pop_results_df.iloc[0]['freq'] < freq_range[1]))
        x_L_power_hist_pop = [p['x_L_power'][freq_indices] for i, p in control_pop_results_df.iteritems() if
                              p['x_L_power'] is not np.NaN and len(p['x_L_power']) >= max(freq_indices[0])]
        _ = ax3.hist(np.log10(np.concatenate(x_L_power_hist_pop, axis=0)), 50, color='gray', histtype='stepfilled',
                     edgecolor='gray', alpha=.25)

        x_R_power_hist_pop = [p['x_R_power'][freq_indices] for i, p in control_pop_results_df.iteritems() if
                              p['x_R_power'] is not np.NaN and len(p['x_R_power']) >= max(freq_indices[0])]
        _ = ax4.hist(np.log10(np.concatenate(x_R_power_hist_pop, axis=0)), 50, color='gray', histtype='stepfilled',
                     edgecolor='gray', alpha=.25)

    if len(f) == len(x_R_power) == len(x_L_power):

        plt.plot(f, np.log10(x_L_power), c=np.array(C[0]) / 2 + .5, label='Left eye', lw=3, alpha=1)
        plt.plot(f, np.log10(x_R_power), c=np.array(C[1]) / 2 + .5, label='Right eye', lw=3, alpha=1)
        plt.xlim([0, 43])
        ylimits = plt.ylim()
        plt.gca().add_patch(
            plt.Rectangle((freq_range[0], ylimits[0] + .15), freq_range[1] - freq_range[0], 0.25, color='k', alpha=0.3,
                          fill=True))
        plt.legend()
    else:
        # todo: why does this ever happen?
        raise Warning("Convergence: Frequency not same length as power, f: " + str(len(f)) + \
                      ", power_L" + str(len(x_L_power)) + \
                      ", power_R: " + str(len(x_R_power)))

    # plot PSD's for beta; left and right eye
    ax3a = ax3.twinx()
    x_L_power_hist_cur = processed_data['x_L_power'][(processed_data['freq'] > 15) & (processed_data['freq'] < 30)]
    _ = ax3a.hist(np.log10(x_L_power_hist_cur), histtype='stepfilled', color=C[0] / 2 + .5, edgecolor=C[0])

    ax4a = ax4.twinx()
    x_R_power_hist_cur = processed_data['x_R_power'][(processed_data['freq'] > 15) & (processed_data['freq'] < 30)]
    _ = ax4a.hist(np.log10(x_R_power_hist_cur), histtype='stepfilled', color=C[1] / 2 + .5, edgecolor=C[1])

    # plot spectrogram and eye position for left eye
    ax5a = fig.add_subplot(spec[0:2, 2])
    plt.pcolormesh(t_spec, f_spec, np.log10(Sxx_L_spec))
    plt.ylim(0, f_spec[-1])
    plt.text(0.02, 0.88, 'Left eye', fontsize=12, fontweight='bold', c='w', transform=ax5a.transAxes)
    ax5b = ax5a.twinx()
    plt.plot(np.linspace(t_spec[0], t_spec[-1], x_L.shape[0]), x_L, 'w')
    plt.xlim(t_spec[0], t_spec[-1])

    # plot spectrogram and eye position for right eye
    ax6a = fig.add_subplot(spec[2:4, 2])
    plt.pcolormesh(t_spec, f_spec, np.log10(Sxx_R_spec))
    plt.ylim(0, f_spec[-1])
    plt.text(0.02, 0.88, 'Right eye', fontsize=12, fontweight='bold', c='w', transform=ax6a.transAxes)
    ax6b = ax6a.twinx()
    plt.plot(np.linspace(t_spec[0], t_spec[-1], x_R.shape[0]), x_R, 'w')
    plt.xlim(t_spec[0], t_spec[-1])
    # nicefy(fsize=10)

    return



def plot_selfpacedsaccade(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    processed_data = current_df.iloc[0]['processed']

    current_results = pd.Series(processed_data)

    # separate right and left side of current subject
    right_side_subj = []
    left_side_subj = []
    # separate right and left side of control population
    right_side_pop = []
    left_side_pop = []

    for value in current_results['data']['saccades']['saccades']:
        if (np.real(current_results['data']['eye'][value[-1]]) >= 0):
            right_side_subj.append(current_results['data']['eye'][value[-1]])
        else:
            left_side_subj.append(current_results['data']['eye'][value[-1]])

    if control_df is not None:
        control_pop_results_df = control_df['processed']

        for i in range(len(control_pop_results_df)):
            row = control_pop_results_df.iloc[i]
            for value in row['data']['saccades']['saccades']:
                if (np.real(row['data']['eye'][value[-1]]) >= 0):
                    right_side_pop.append(row['data']['eye'][value[-1]])
                else:
                    left_side_pop.append(row['data']['eye'][value[-1]])

        # Calculate bins used for both control and subject using control population
        # (we want the bins to be shared to directly compare the two)
        left_pop, left_bins = np.histogram(np.real(left_side_pop), bins=30, density=True)
        right_pop, right_bins = np.histogram(np.real(right_side_pop), bins=30, density=True)
    else:
        control_pop_results_df = pd.DataFrame()

        # Calculate bins used for both control and subject using subject
        # (we want the bins to be shared to directly compare the two)
        left_subj, left_bins = np.histogram(np.real(left_side_subj), bins=30, density=True)
        right_subj, right_bins = np.histogram(np.real(right_side_subj), bins=30, density=True)

    # convert left and right to array so we can use plotting functions later
    right_side_pop = np.array(right_side_pop)
    left_side_pop = np.array(left_side_pop)
    right_side_subj = np.array(right_side_subj)
    left_side_subj = np.array(left_side_subj)

    # Plot both sides of control population and current
    # subject on scatter/contour plot and histogram
    fig_sizer(10)
    fig = plt.figure(constrained_layout=True)

    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[2, 1])
    ax_eye = fig.add_subplot(spec[0, 0])
    ax_eye_x = fig.add_subplot(spec[1, 0])

    plt.sca(ax_eye)

    # if the control_df is None, ndhist will fail so we do this check
    # more specifically if left_side_pop or right_side_pop are empty
    # this plots the contours of the population data
    if (len(left_side_pop) != 0) and (len(right_side_pop) != 0):
        _ = ndhist(left_side_pop, markertype=None, fx=10, fy=3, smooth=2, levels=[20, 50])
        _ = ndhist(right_side_pop, markertype=None, fx=10, fy=3, smooth=2, levels=[20,50])
    cplot(left_side_subj, 'o', c=C[1], label='left_side_subject')

    cplot(right_side_subj, 'o', c=C[0], label='right_side_subject')
    xlim([-2, 2])
    ylim([-0.5, 0.5])
    legend()
    plt.title("Self-Paced Saccade Eye Movements")
    nicefy()

    plt.sca(ax_eye_x)

    plt.hist(np.real(left_side_pop), bins=left_bins, label='control', color='gray', density=True, alpha=0.3)
    sns.kdeplot(np.real(left_side_pop), bw=0.2, color=[0.7,0.7,0.7])
    plt.hist(np.real(left_side_subj), bins=left_bins, label='left_side_subject', color=C[1], density=True, alpha=0.5)
    sns.kdeplot(np.real(left_side_subj), bw=0.2, color=C[1])

    plt.hist(np.real(right_side_pop), bins=right_bins, color='gray', density=True, alpha=0.3)
    sns.kdeplot(np.real(right_side_pop), bw=0.2, color='gray')
    plt.hist(np.real(right_side_subj), bins=right_bins, label='right_side_subject', color=C[0], density=True, alpha=0.5)
    sns.kdeplot(np.real(right_side_subj), bw=0.2, color=C[0])

    xlim([-2, 2])
    legend(loc='upper center', fontsize=12)
    plt.title("X-distribution of Eye Movements")
    nicefy()

    return

