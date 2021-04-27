def get_exam_vizs_debug():

    trial_vizs = OrderedDict()

    trial_vizs['prosaccade'] = [plot_prosaccade_xy,
                                plot_ds,
                                plot_instructions]

    trial_vizs['gaze'] = [plot_gaze,
                          blink_detection,
                          plot_ds,
                          plot_instructions]

    trial_vizs['smoothpursuit2d'] = [plot_smoothpursuit2d,
                                     blink_detection,
                                     plot_ds,
                                     plot_instructions]

    trial_vizs['pupillaryreflex'] = [plot_pupillaryreflex,
                                     blink_detection,
                                     plot_ds,
                                     plot_instructions]
    
    trial_vizs['selfpacedsaccade'] = [plot_selfpaced_count_short,
                                      plot_selfpaced_count_long,
                                      plot_selfpaced_velocity_long,
                                      plot_selfpaced_velocity_long_hist,
                                      blink_detection,
                                      plot_ds,
                                      plot_instructions]

    trial_vizs['smoothpursuit'] = [plot_smoothpursuit,
                                   blink_detection,
                                   plot_ds,
                                   plot_instructions]

    trial_vizs['convergence'] = [plot_convergence,
                                 blink_detection,
                                 plot_ds,
                                 plot_instructions]

    trial_vizs['vor'] = [plot_vor,
                         plot_vor_xcorr,
                         blink_detection,
                         plot_ds,
                         plot_instructions]

    return trial_vizs


def get_exam_vizs():

    trial_vizs = OrderedDict()

    trial_vizs['prosaccade'] = [plot_prosaccade_xy]

    trial_vizs['gaze'] = [plot_gaze]

    trial_vizs['smoothpursuit2d'] = [plot_smoothpursuit2d]

    trial_vizs['pupillaryreflex'] = [plot_pupillaryreflex]

    trial_vizs['selfpacedsaccade'] = [plot_selfpaced_count_short,
                                      plot_selfpaced_count_long,
                                      plot_selfpaced_velocity_long,
                                      plot_selfpaced_velocity_long_hist]

    trial_vizs['smoothpursuit'] = [plot_smoothpursuit]

    trial_vizs['convergence'] = [plot_convergence]

    trial_vizs['vor'] = [plot_vor,
                         plot_vor_xcorr]

    trial_vizs['vorx'] = [dummy]

    trial_vizs['periphery'] = [dummy]

    trial_vizs['trailmaking'] = [plot_trailmaking]

    trial_vizs['trailmaking2'] = [plot_trailmaking]

    trial_vizs['categoryfluency'] = [dummy]

    return trial_vizs

    def plot_selfpaced_count_long(df, cindex, pindex):
    """
    Process and visualize self guided saccade digital exam
    :param df: dataframe with visit_exam_id and timeseries dataframe
           cindex: index of df for visit_exam_id to be processed
           pindex: array-like indices of df representing visit_exam_id
                    of control population
    :return
    """

    # apply analysis to df
    results = df.timeseries.apply(analysis.process_selfpacedsaccade)
    results = results.merge(df['visit_exam_id'], left_index=True, right_index=True)

    # process control population
    results['population'] = results.apply(lambda x: categorize_population_by_visit_exam_id(x,
                                                                                            cindex,
                                                                                            pindex),
                                            axis=1)

    # compute cdf to map colors based on population data
    p_num_long_saccades = results.loc[results['population']=='control', 'num_long_saccades']
    c_num_long_saccades = results.loc[results['population']=='subject', 'num_long_saccades']

    hist, edges = np.histogram(p_num_long_saccades, bins=50)
    cdf = np.cumsum(hist)
    cdf = cdf/np.max(cdf)

    # color mappings by value specific to this digital exam
    color_change = [0, # min saccades
                    edges[np.where(cdf > .05)[0][0]], # probability
                    edges[np.where(cdf > .15)[0][0]], # probability
                    edges[np.where(cdf > .25)[0][0]], # probability
                    edges[np.where(cdf > .99)[0][0]]] # max saccades

    color_list = ['red',
                  'yellow',
                  'green',
                  'green',
                  'green']

    cmap = custom_cmap(color_change, color_list)
    view_colormap(color_change, cmap)
    plt.scatter(c_num_long_saccades, .1, s=300, marker='v', color='k')
    plt.xlabel('Number of Long Saccades')
    nicefy()


def plot_selfpaced_count_short(df, cindex, pindex):
    """
    Process and visualize self guided saccade digital exam
    :param df: dataframe with visit_exam_id and timeseries dataframe
           cindex: index of df for visit_exam_id to be processed
           pindex: array-like indices of df representing visit_exam_id
                    of control population
    :return
    """

    # apply analysis to df
    results = df.timeseries.apply(analysis.process_selfpacedsaccade)
    results = results.merge(df['visit_exam_id'], left_index=True, right_index=True)

    # process control population
    results['population'] = results.apply(lambda x: categorize_population_by_visit_exam_id(x,
                                                                                            cindex,
                                                                                            pindex),
                                            axis=1)

    # compute cdf to map colors based on population data
    p_num_short_saccades = results.loc[results['population']=='control', 'num_short_saccades']
    c_num_short_saccades = results.loc[results['population']=='subject', 'num_short_saccades']

    hist, edges = np.histogram(p_num_short_saccades, bins=50)
    cdf = np.cumsum(hist)
    cdf = cdf/np.max(cdf)

    # color mappings by value specific to this digital exam
    color_change = [0, # min saccades
                    edges[np.where(cdf > .05)[0][0]], # probability
                    edges[np.where(cdf > .15)[0][0]], # probability
                    edges[np.where(cdf > .25)[0][0]], # probability
                    edges[np.where(cdf > .99)[0][0]]] # max saccades

    color_list = ['red',
                  'yellow',
                  'green',
                  'green',
                  'green']

    cmap = custom_cmap(color_change, color_list)
    view_colormap(color_change, cmap)
    plt.scatter(c_num_short_saccades, .10, s=300, marker='v', color='k')
    plt.xlabel('Number of short saccades')
    nicefy()


def plot_selfpaced_velocity_long_hist(df, cindex, pindex):
    """
    Process and visualize self guided saccade digital exam
    :param df: dataframe with visit_exam_id and timeseries dataframe
           cindex: index of df for visit_exam_id to be processed
           pindex: array-like indices of df representing visit_exam_id
                    of control population
    :return
    """

    # apply analysis to df
    results = df.timeseries.apply(analysis.process_selfpacedsaccade)
    results = results.merge(df['visit_exam_id'], left_index=True, right_index=True)

    # process control population
    results['population'] = results.apply(lambda x: categorize_population_by_visit_exam_id(x,
                                                                                            cindex,
                                                                                            pindex),
                                            axis=1)

    # compute cdf to map colors based on population data
    p_long_velocity = results.loc[results['population']=='control', 'long_saccades_velocity']
    c_long_velocity = results.loc[results['population']=='subject', 'long_saccades_velocity']

    fig,ax = plt.subplots(figsize=(8,5), tight_layout=True)
    for i in range(p_long_velocity.size):
        sns.distplot(p_long_velocity.iloc[i], hist=False, color='k', kde_kws={'alpha': .1})
        ax.set_xlim(0, 3000)
        ax.set_yticks([])

    sns.distplot(c_long_velocity[0], hist=False, color='r', kde_kws={'linewidth':3, 'alpha': 1})
    ax.set_xlim(0, 3000)
    ax.set_yticks([])
    ax.set_xlabel('Long saccade velocity')
    nicefy()


def plot_selfpaced_velocity_long(df, cindex, pindex):
    """
    Process and visualize self guided saccade digital exam
    :param df: dataframe with visit_exam_id and timeseries dataframe
           cindex: index of df for visit_exam_id to be processed
           pindex: array-like indices of df representing visit_exam_id
                    of control population
    :return
    """

    # apply analysis to df
    results = df.timeseries.apply(analysis.process_selfpacedsaccade)
    results = results.merge(df['visit_exam_id'], left_index=True, right_index=True)

    # process control population
    results['population'] = results.apply(lambda x: categorize_population_by_visit_exam_id(x,
                                                                                            cindex,
                                                                                            pindex),
                                            axis=1)

    # compute cdf to map colors based on population data
    p_long_velocity = results.loc[results['population']=='control', 'long_saccades_velocity']
    c_long_velocity = results.loc[results['population']=='subject', 'long_saccades_velocity']
    p_long_velocity = p_long_velocity.apply(np.median).values
    c_long_velocity = c_long_velocity.apply(np.median).values

    # filter fast saccades - likely erroneous
    p_long_velocity = p_long_velocity[p_long_velocity<2000]

    hist, edges = np.histogram(p_long_velocity, bins=50)
    cdf = np.cumsum(hist)
    cdf = cdf/np.max(cdf)

    # color mappings by value specific to this digital exam
    color_change = [0,
                    edges[np.where(cdf > .25)[0][0]],
                    edges[np.where(cdf > .30)[0][0]],
                    edges[np.where(cdf > .99)[0][0]]]

    color_list = ['red',
                  'yellow',
                  'green',
                  'green']

    cmap = custom_cmap(color_change, color_list)
    view_colormap(color_change, cmap)
    plt.scatter(c_long_velocity, .10, s=300, marker='v', color='k')
    plt.xlabel('Median saccade velocity')
    nicefy()

def plot_prosaccade_timeseries(df, xy='X'):

    mult_fac = {'X': -1,
                'Y': 1}
    x = df['Total Time'].values
    # y1 = df['Dot1 Position ' + xy].values
    y1 = utils_df.robust_get_stimulus(df, xy, '1').values

    y2 = df['Left Eye Position ' + xy].values

    I = (y1 != 0) & (y2 != 0)
    #     fig, ax1 = plt.subplots()
    ax1 = gca()
    ax1.plot(x[I], y1[I], '.', color=C[0])
    ax1.plot(x[I], y1[I], '--', color=C[0])
    ax1.set_ylabel('Stim 1 Pos ' + xy, color=C[0])

    ax2 = ax1.twinx()
    ax2.plot(x[I],mult_fac[xy] * y2[I], color=C[1])
    ax2.set_ylabel('Left Eye Pos ' + xy, color=C[1])
    nicefy()


def plot_prosaccade_reaction(df_or_complete_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(df_or_complete_df, visit_exam_id)


    all_reaction_times = []
    for idx, cur_row in control_df.iterrows():
        result_n = analysis.process_prosaccade(cur_row['timeseries'])
        all_reaction_times += result_n['reaction_times']

    df = df_or_complete_df.loc[df_or_complete_df.visit_exam_id == visit_exam_id].iloc[0]['timeseries']
    user_id = df_or_complete_df.loc[df_or_complete_df.visit_exam_id == visit_exam_id].iloc[0]['user_id']


    result = analysis.process_prosaccade(df)

    to_plot = {'control': all_reaction_times, user_id: result['reaction_times']}
    nhist(to_plot, xlabel='reaction times', normalize='frac', same_bins_flag=True)
    nicefy()

    return 'png'


def plot_prosaccade_xy(df, cindex):

    df = df.loc[df.visit_exam_id == cindex].iloc[0]
    data = df.timeseries

    # C = linspecer(4)
    mult_fac = {'X': -1,
                'Y': 1}

    xy2c = {'X':0, 'Y': 1}
    for xy in ['X', 'Y']:
        x = data['Total Time'].values
        # y1 = data['Dot1 Position ' + xy].values
        y1 = utils_df.robust_get_stimulus(data, xy, '1').values

        y2 = data['Left Eye Position ' + xy].values

        I = (y1 != 0) & (y2 != 0)
        #     fig, ax1 = plt.subplots()
        ax1 = gca()
        # ax1.plot(x[I], y1[I], '.', color=C[xy2c[xy]])
        ax1.plot(x[I], y1[I], '--', color=C[xy2c[xy]])
        # ax1.set_ylabel('Stimulus Position', color=C[0])
        gca().set_yticks([])
    legend(['stimulus position x', 'stimulus position y'], loc='lower left')

    # legend(['X eye', "Y eye" ])
    for xy in ['X', 'Y']:
        x = data['Total Time'].values
        # y1 = data['Dot1 Position ' + xy].values
        y1 = utils_df.robust_get_stimulus(data, xy, '1').values

        y2 = data['Left Eye Position ' + xy].values

        I = (y1 != 0) & (y2 != 0)

        ax2 = ax1.twinx()
        ax2.plot(x[I], mult_fac[xy] * y2[I], color=C[xy2c[xy]])
        # ax2.set_ylabel('Left Eye Pos ' + xy, color=C[xy2c[xy]])
    legend(['eye position X', 'eye position Y'], loc="lower right")

    nicefy()

def replay_setup_figure(subj_series):

    fig = plt.figure(figsize=(6,4), constrained_layout=False)
    # fig.set_visible(False)
    gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0)
    eye_fig = fig.add_subplot(gs[:-1, :])
    sub_fig = fig.add_subplot(gs[-1, :])

    leye, = eye_fig.plot([], [], 'b-o')
    reye, = eye_fig.plot([], [], 'r-o')

    eye_fig.set_xlim(-1, 1)
    eye_fig.set_ylim(-1, 1)
    eye_fig.set_xticks([])
    eye_fig.set_yticks([])

    sub_fig.set_xticks([])
    sub_fig.set_yticks([])
    sub_fig.set_xlim(0, len(subj_series['timeseries']))

    mv.viz.set_fontsize()

    playhead = sub_fig.axvline(x=0, ymin=0, ymax=1)

    handles = {'fig': fig,
               'eye_fig': eye_fig,
               'sub_fig': sub_fig,
               'leye': leye,
               'reye': reye,
               'playhead': playhead}

    return handles


def replay_plot_subfigure(subj_series, handles):

    # map exam_id to subfigure function
    exam2subfig = {4: replay_subfigure_prosaccade,
                   5: replay_subfigure_convergence,
                   6: replay_subfigure_pupilreflex,
                   7: replay_subfigure_gaze,
                   11: replay_subfigure_smoothpursuit2d,
                   10: replay_subfigure_selfpaced,
                   15: replay_subfigure_trailmaking,
                   16: replay_subfigure_trailmaking,
                   }

    # call function
    exam2subfig[subj_series['exam_id']](subj_series, handles)


def replay_subfigure_trailmaking(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')

def replay_subfigure_prosaccade(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')


def replay_subfigure_gaze(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')


def replay_subfigure_smoothpursuit2d(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')


def replay_subfigure_selfpaced(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')

    # twin = handles['sub_fig'].twinx()
    # twin.plot(data['lep'], 'k-', alpha=.2, linewidth=.5)
    # twin.plot(data['rep'], 'k-', alpha=.2, linewidth=.5)
    # twin.set_xticks([])
    # twin.set_yticks([])


def replay_subfigure_pupilreflex(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lep'], 'b', linewidth=1)
    handles['sub_fig'].plot(data['rep'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye Pupil Diameter')



def replay_subfigure_convergence(df, handles):

    data = get_replay_eye_data(df)

    handles['sub_fig'].plot(data['lex'], 'b', linewidth=1)
    handles['sub_fig'].plot(-data['rex'], 'r', linewidth=1)
    handles['sub_fig'].set_xlabel('L/R Eye X Direction')


def get_replay_eye_data(subj_series):

    lex = subj_series.timeseries['Left Eye Direction X']
    ley = subj_series.timeseries['Left Eye Direction Y']
    lep = subj_series.timeseries['Left Pupil Diameter'] * 2

    rex = subj_series.timeseries['Right Eye Direction X']
    rey = subj_series.timeseries['Right Eye Direction Y']
    rep = subj_series.timeseries['Right Pupil Diameter'] * 2

    cex = subj_series.timeseries['Combine Eye Direction X']
    cey = subj_series.timeseries['Combine Eye Direction Y']

    data = {'lex': lex,
            'ley': ley,
            'lep': lep,
            'rex': rex,
            'rey': rey,
            'rep': rep,
            'cex': cex,
            'cey': cey}

    return data


def replay_save(subj_series, handles, path):

    fps = np.round(1 / subj_series.timeseries['Delta Time'].mean())
    writer = FFMpegWriter(fps=fps)

    data = get_replay_eye_data(subj_series)

    with writer.saving(handles['fig'], path, dpi=500):
        for i in range(len(subj_series.timeseries)):

            handles['leye'].set_data(data['lex'][i], data['ley'][i])
            handles['leye'].set_markersize(data['lep'][i])

            handles['reye'].set_data(data['rex'][i], data['rey'][i])
            handles['reye'].set_markersize(data['rep'][i])

            handles['playhead'].set_data([i,i], [0, 1])

            writer.grab_frame()
            
    plt.clf()

    return path


def replay_digital_exam(subj_series, path):

    # setup figure
    handles = replay_setup_figure(subj_series)

    # plot subfigure; specific for each digital exam
    replay_plot_subfigure(subj_series, handles)

    # save video
    replay_save(subj_series, handles, path)

    return path


