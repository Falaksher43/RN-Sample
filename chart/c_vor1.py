from chart.c_general import *


def plot_prosaccade(processed_df, visit_exam_id=None):
    # split processed df into current and control
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)

    # plot prosaccade eye movements and reaction times
    fig_sizer(10, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[2.5, 1])

    ax_eye_short = fig.add_subplot(spec[0, 0])
    ax_eye_long = fig.add_subplot(spec[0, 1])

    ax_react_short = fig.add_subplot(spec[1, 0])
    ax_react_long = fig.add_subplot(spec[1, 1], sharey=ax_react_short)

    plot_prosaccade_movements(current_df, ax_eye_short, ax_eye_long, threshold=3)
    plot_prosaccade_reaction_times(current_df, control_df, ax_react_short, ax_react_long, threshold=.22)

    return 'png'


def plot_prosaccade_movements(current_df, ax_eye_short, ax_eye_long, threshold=3):

    # convert current_df into sdf for plotting
    processed_data = current_df.iloc[0]['processed']['data']
    sdf = analysis.p_results_to_sdf(processed_data)

    # plotting eye movements
    # todo: need to set this threshold properly, currently arbitrary
    for index, row in sdf.iterrows():
        if (len(row['eye_timeseries']) == 0) | (len(row['t_timeseries']) == 0):
            continue

        if row['saccade_count_per_path'] <= threshold:
            cur_c = [0, 0.65, 0]
        elif row['saccade_count_per_path'] == threshold + 1:
            cur_c = [1, 0.7, 0]
        else:
            cur_c = [1, 0, 0]

        #     pick the right subplots
        if row['distance'] == 0.5:
            cur_ax = ax_eye_short
        else:
            cur_ax = ax_eye_long

        #       plot the eye movements
        plt.sca(cur_ax)
        plot_prosaccade_eye_speed(row, cur_c)

    plt.sca(ax_eye_short)
    title('short saccade')
    prep_axis()
    xylim([-.8, .8])

    plt.sca(ax_eye_long)
    title('long saccade')
    prep_axis()
    xylim([-1.5, 1.5])


def plot_prosaccade_reaction_times(current_df, control_df, ax_react_short, ax_react_long, threshold=0.22):
    # used to plot reaction times for each stim_vec in its own column
    def stim_vec_to_x(stim_vec):
        """Used to position bar plots and points"""
        idx = stim_vecs.index(stim_vec)
        if idx > 5:
            idx = idx - 6
        return idx

    # stim_vecs associated with prosaccade exam
    # order is important here for plotting indices later
    stim_vecs = [-0.5 + 0.j, -0.25 + 0.433j, 0.25 + 0.433j, 0.5 + 0.j, 0.25 - 0.433j, -0.25 - 0.433j,
                 -1. + 0.j, -0.5 + 0.866j, 0.5 + 0.866j, 1. + 0.j, 0.5 - 0.866j, -0.5 - 0.866j]

    # extract processed prosaccade data from overall dataframes and convert into sdf
    processed_data = current_df.iloc[0]['processed']['data']
    sdf = analysis.p_results_to_sdf(processed_data)

    if control_df is not None:
        control_pop_results_df = control_df['processed']
        num_controls = len(control_pop_results_df)

        # aggregating all of the reaction times from controls
        all_reaction_times = {w: [] for w in stim_vecs}
        control_results = []
        for series in control_pop_results_df:
            result_n = series['data']
            if len(result_n['reaction_times']) == 0:
                continue
            try:
                sdf_n = analysis.p_results_to_sdf(result_n)
            except:
                logger.debug("some error calculating saccades")
                continue

            for idx, cur in sdf_n.iterrows():
                all_reaction_times[cur['stim_vec']].append(cur['reaction_times'])

            react_std = np.std(result_n['reaction_times'])
            control_results.append({'number of saccades': result_n['number of saccades'],
                                    'reaction time median': result_n['median reaction time'],
                                    'reaction time std': react_std,
                                    'reaction time median short': np.nanmedian(
                                        sdf_n.loc[sdf_n.distance == 0.5, 'reaction_times']),
                                    'reaction time median long': np.nanmedian(
                                        sdf_n.loc[sdf_n.distance == 1, 'reaction_times'])})

        control_results = pd.DataFrame(control_results)
        # barplot the control data
        for idx, stim_vec in enumerate(stim_vecs):
            if idx <= 5:
                plt.sca(ax_react_short)
            else:
                plt.sca(ax_react_long)

            reactions = all_reaction_times[stim_vec]
            if len(reactions) > 0:
                y = np.median(reactions)
                yerr = np.std(reactions) / np.sqrt(len(reactions))
                create_single_bar(y, yerr, stim_vec_to_x(stim_vec))
                ylabel("median reaction times ± σ")

        # adding combined reaction times for all directions
        all_reactions_short = []
        all_reactions_long = []
        for idx, stim_vec in enumerate(stim_vecs):
            if idx <= 5:
                all_reactions_short.extend(all_reaction_times[stim_vec])
            else:
                all_reactions_long.extend(all_reaction_times[stim_vec])

        reaction_data = control_results[['reaction time median short', 'reaction time median long']]
    else:
        control_results = pd.DataFrame()
        all_reaction_times = []
        reaction_data = pd.DataFrame()
        num_controls = 0

    # todo: completely arbitrary threshold, need to set this properly
    for index, row in sdf.iterrows():
        if (len(row['eye_timeseries']) == 0) | (len(row['t_timeseries']) == 0):
            continue

        if row['reaction_times'] <= threshold:
            cur_c = [0, 0.65, 0]
        else:
            cur_c = [1, 0, 0]

        #     pick the right subplots
        if row['distance'] == 0.5:
            cur_ax = ax_react_short
        else:
            cur_ax = ax_react_long

        #       plot reaction times
        plt.sca(cur_ax)
        plot([stim_vec_to_x(row['stim_vec']) + .2 * (np.random.rand() - 0.5)], [row['reaction_times']], 'o',
             markersize=11, c=cur_c, markeredgewidth=1, alpha=.5)

    axes = [ax_react_short, ax_react_long]
    for index, (axes, data) in enumerate(zip(axes, reaction_data.iteritems())):

        if len(reaction_data) > 0:
            plt.sca(axes)
            color = 'gray'
            gca().boxplot(data[1],
                          notch=False,
                          patch_artist=True,
                          boxprops=dict(color=color, facecolor=color, alpha=.2, lw=2),
                          capprops=dict(color=color, alpha=.8),
                          whiskerprops=dict(color=color, alpha=.8),
                          positions=[6.0],
                          widths=.3,
                          showfliers=False)

        # plot current data
        if index == 0:
            c_reaction_data = sdf.loc[sdf.distance == 0.5, 'reaction_times']
        else:
            c_reaction_data = sdf.loc[sdf.distance == 1.0, 'reaction_times']

        color = 'blue'
        gca().boxplot(c_reaction_data,
                      notch=False,
                      patch_artist=True,
                      boxprops=dict(color=color, facecolor=color, alpha=.2),
                      capprops=dict(color=color, alpha=.8),
                      whiskerprops=dict(color=color, alpha=.8),
                      positions=[6.5],
                      widths=.3,
                      showfliers=False)

        ylabel("median reaction times ± σ")
        xlim([-.5, 7.0])
        xticks(range(7), ['←', '↖', '↗', '→', '↘', '↙', 'all'], fontsize=15)

    plt.sca(ax_react_long)
    title("long saccades, controls = " + str(num_controls))
    nicefy()

    plt.sca(ax_react_short)
    title("short saccades, controls = " + str(num_controls))
    nicefy()

# used to save the parameter settings for both short and long prosaccade saccade charts
def plot_prosaccade_eye_speed(row, c):
    plot_eye_speed(row['eye_timeseries'] - row['eye_fixs'] + row['stim_vec'], row['t_timeseries'], c=c)
    cplot(row['stim_vec'], 'ok', markersize=11, fillstyle='none', markeredgewidth=2, label='_nolegend_')


def plot_smoothpursuit(processed_df, visit_exam_id):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    result = current_df.iloc[0]['processed']

    controls_exist = control_df is not None
    # if controls_exist:
        # lags_control = np.concatenate([w['data']['lags_all']['left'] for w in control_df['processed'].values])
    # else:
    #     lags_control = []

    t = result['data']['t']
    # x = result['data']['x']
    L = result['data']['L']
    R = result['data']['R']
    stim = result['data']['stim']

    subplotter(3, 2, [0, 1])
    plot(t, stim, '--k')
    plot(t, L, color=C[0])
    plot(t, R, color=C[1])
    ylabel('Eye Position')
    legend(['stim', 'Left', 'Right'])
    xlabel('Time (s)')

    subplotter(3, 2, [2, 3])
    plot(t, L - stim, color=C[0])
    plot(t, R - stim, color=C[1])
    legend(['Left', 'Right'])
    ylabel('Eye lag')

    eyes = [0, 0, 0, 0, 'left', 'right']
    for idx in [4, 5]:
        subplotter(3, 2, idx)
        if controls_exist:
            lags_control = np.concatenate([w['data']['lags_all'][eyes[idx]] for w in control_df['processed'].values])
            to_hist = {'control': lags_control, 'subject': result['data']['lags_all'][eyes[idx]]}
        else:
            to_hist = result['data']['lags_all'][eyes[idx]]
        nhist(to_hist, noerror=True, normalize='frac')
        xlabel(eyes[idx] + ' eye lags (s)')
        format_axis_date()

    nicefy()



def prep_axis():
    """Used for every eye chart, prepare the axis"""
    axis('equal')
    cplot(0, '.k', markersize=15)
    axis('off')
    nicefy(fsize=12)


def plot_pupillaryreflex(processed_df, visit_exam_id=None):
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    processed_data = current_df.iloc[0]['processed']

    # run analyses on population, if present
    if control_df is not None:
        control_pop_results_df = control_df['processed']
        num_controls = len(control_pop_results_df)
    else:
        control_pop_results_df = None
        num_controls = 0

    def plot_bkg(t, stim_idx):
        yy = ylim()
        plot_range([0, max(t)], y_offset=yy[0], height=yy[1] - yy[0], color=[.8, .8, .8])
        plot_range([t[stim_idx[0]], t[stim_idx[1]]], y_offset=yy[0], height=yy[1] - yy[0], color='white')
        plot_range([t[stim_idx[0]], t[stim_idx[1]]], y_offset=yy[0], height=yy[1] - yy[0], color='white')
        plot_range([t[stim_idx[0]], t[stim_idx[1]]], y_offset=yy[0], height=yy[1] - yy[0], color='white')
        plot_range([t[stim_idx[0]], t[stim_idx[1]]], y_offset=yy[0], height=yy[1] - yy[0], color='white')

    current_results = pd.Series(processed_data)
    control_results = control_pop_results_df

    fig_sizer(8, 10)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, height_ratios=[1.75, 1])

    ax2 = fig.add_subplot(spec[0, :])

    if control_results is not None:
        for index, results in control_results.iteritems():
            plot(results['data']['t'], np.imag(results['data']['pupil']), c='gray', marker='.', markersize=2, alpha=.2)

    plot(current_results['data']['t'], np.real(current_results['data']['pupil']), 'r.', lw=1.5)
    plot(current_results['data']['t'], np.imag(current_results['data']['pupil']), 'b.', lw=1.5)

    plot_bkg(current_results['data']['t'], current_results['data']['stim_start_end'])

    ylabel('Normalized Pupil Diameter (mm)')
    xlabel('Time (s)')
    title('Controls n=' + str(num_controls))
    nicefy()

    ax3 = fig.add_subplot(spec[1, 0])
    y = 1  # this is just a hardcoded value in case there are no controls
    if control_results is not None:
        range_data = [w['metrics']['pupil_range'] for w in control_results]
        _, n, _ = nhist(range_data, f=3, color='gray', noerror=True)
        y = max(n[0])
    error_subj = current_results['metrics']['pupil_range']
    plot([error_subj, error_subj], [0, y], linewidth=3, color=C[0])
    plot([error_subj], [y], 'o', color=C[0], markersize=15)
    title('pupil range')
    ylim([0, y * 1.1])
    format_axis_date()
    nicefy()

    # todo: check if we are trying to plot pupil_dilation or constriction
    ax4 = fig.add_subplot(spec[1, 1])
    if control_results is not None:
        velocity_data = [w['metrics']['pupil_dilation_velocity'] for w in control_results]
        _, n, _ = nhist(velocity_data, f=3, color='gray', noerror=True)
        y = max(n[0])
    error_subj = current_results['metrics']['pupil_dilation_velocity']
    plot([error_subj, error_subj], [0, y], linewidth=3, color=C[0])
    plot([error_subj], [y], 'o', color=C[0], markersize=15)
    title('constriction velocity')
    format_axis_date()
    ylim([0, y * 1.1])
    nicefy()

    ax5 = fig.add_subplot(spec[1, 2])
    # note: randomly sorting this for a nice mix of folks
    num_LR_points_to_compare = 20000
    if control_results is not None:
        pupil_pop = np.random.permutation(np.hstack([w['data']['pupil'] for w in control_results]))
        # safely only select
        _ = ndhist(pupil_pop[:num_LR_points_to_compare], smooth=1, levels=[50, 85, 95])
    # cplot(current_results['data']['pupil'], '.', markersize=3, c=C[0])
    logfit(current_results['data']['pupil'], graph_type='linear',
           marker_style={'marker': '.', 'linestyle': 'none', 'markersize': 4, 'c': C[0]},
           line_style={'c': 'k', 'linestyle': '-', 'lw': 1})
    plot_diag(lw=4)
    title('assymetric pupil')
    xlabel('Right pupil')
    ylabel('Left pupil')
    format_axis_date()
    nicefy()



