from chart.c_general import *



def plot_trail_accuracy(cur_result):
    # when you first land on the right target
    style_correct = {'color': 'g',
                     'marker': '.',
                     'markeredgecolor': 'g',
                     'markersize': 13,
                     'markerfacecolor': 'None',
                     'linewidth': 0,
                     'markeredgewidth': 3}

    # when someone goes back to the same one they were on before
    style_rehover = {'color': 'b',
                   'marker': 'o',
                   'linewidth': 0,
                   'markeredgecolor': 'None'}

    # when you bump into the wrong target
    style_wrong = {'color': 'r',
                   'marker': 'x',
                   'markeredgewidth': 3,
                   'linewidth': 0,
                    'markersize': 13}

    # black is when you hover for longer
    style_neutral = {'color': 'k',
                     'marker': '.'}

    # make a couple charts just for the legend
    for cur_style in [style_correct, style_neutral, style_wrong, style_rehover]:
        plot([-10, -10],[2, 2], **cur_style)

    legend(['correct target', 'thinking', 'miss', 'same target'], loc='best')


    selected_index = cur_result['selected index']
    stop_labels = cur_result['stop labels']
    stop_times = cur_result['stop times']

    starts_ends = etl.start_and_ends(selected_index)
    time = cur_result['time']

    # stop_values, stop_labels, stop_times, stop_durations, _ = analysis.index_to_labeled_stops(df)

    plot(time, selected_index, '.', color='k')

    for idx in range(len(starts_ends)):
        start, endd = starts_ends[idx]

        if stop_labels[idx] == 0:
            cur_style = style_correct
            y_fac = 1

        elif stop_labels[idx] == 1:
            cur_style = style_rehover
            y_fac = 0

        else:
            cur_style = style_wrong
            y_fac = 0


        plot(time[start+1], selected_index[start+1] - y_fac, **cur_style)

    nicefy()
    xlim(time[0], stop_times[-1] + 3)
    ylabel('index')
    xlabel('time')





def get_normed_error(df):
    """
    Compute the error to target normalized by
    """
#     get the first row for each index (to get the pointer location)
    df2 = df.groupby('new index').first()
#     get the distance from pointer start to the target stimulus
    distance_to_stim = np.abs(df2['new stim'] - df2['pointer']).rename('stim dist')
#     add the static distance to stimulus target to the DF (to normalize)
    df3 = df.join(distance_to_stim, on='new index', how='left')
#
    df3['error'] = np.sqrt(np.abs(df3['new stim'] - df3['pointer'])) / np.sqrt(df3['stim dist'])
    return df3[['new index', 'time', 'error']]



def get_control_distances(processed):
    #     check for good or bad data
    I = processed.apply(lambda result: result['has_error'] == False and result['good data'] == True)
    #     the time per stim for each control
    completion_distance_series = processed[I].apply(get_distance_per_stim)
    #     compute the median
    return completion_distance_series

def get_control_durations(processed):
#     check for good or bad data
    I = processed.apply(lambda result: result['has_error'] == False and result['good data'] == True)
#     the time per stim for each control
    completion_times_series = processed[I].apply(get_duration_per_stim)
#     compute the median
    return completion_times_series

def get_duration_per_stim(result):
    #     stimulus = result['stimulus']
    #     pointer = result['pointer']
    time = result['time']
    new_index = get_new_index(result['selected index'])

    df = pd.DataFrame({'new index': new_index, 'selected index': result['selected index'], 'time': time})

    time_by_stim_start = df.groupby('new index').first()['time']
    time_by_stim_end = df.groupby('new index').last()['time']

    return time_by_stim_end - time_by_stim_start

def get_distance_per_stim(result):
    stimulus = result['stimulus']
    pointer = result['pointer']

    new_index = get_new_index(result['selected index'])

    df = pd.DataFrame(
        {'new index': new_index, 'selected index': result['selected index'], 'stimulus': stimulus, 'pointer': pointer})
    # index_to_stimulus = etl.dictify_cols(df[['selected index', 'stimulus']])

    #     calculate the distance travelled (no smoothing, and no cutting off-the-world movements)
    distance_per_stim = df.groupby('selected index').apply(lambda w: np.sum(np.abs(np.diff(w['pointer']))))

    return distance_per_stim


def get_new_index(selected_index):
    """
    new_index represnts the index of the target currently being searched for.
    """
    new_index = []
    cur_index = 0
    for jj in range(len(selected_index)):

        if selected_index[jj] == cur_index + 1:
            cur_index += 1
            new_index.append(cur_index)

        else:
            new_index.append(cur_index)
    new_index = np.array(new_index)
    return new_index


def plot_distance(cur_result, control_df):
    C = get_subj_control_colors()


    if control_df is not None:
        control_distances = get_control_distances(control_df['processed'])
        control_dist_flat = (control_distances - control_distances.median()).iloc[:, 1:].values.flatten()
        cur_dist_flat = (get_distance_per_stim(cur_result) - control_distances.median()).values[1:]
        _, n, _ = nhist([cur_dist_flat, control_dist_flat],  # legend=['subject', 'control'],
                        maxx=600, minx=-150, noerror=True,
                        same_bins_flag=True, f=1, color=C, normalize='frac', legend=['', ''],
                        )
        y = 1.1 * max(n[0])  # arbitrary height, just for plotting

        # plot the median=0 for the control
        x = 0
        plot_pin(x, y, color='gray')

    else:
        cur_dist_flat = get_distance_per_stim(cur_result)
        _, n, _ = nhist(cur_dist_flat,  # legend=['subject', 'control'],
                        maxx=600, minx=-150, noerror=True,
                        same_bins_flag=True, f=20, color=C, normalize='num', legend=['', ''])

        y = 1.1 * max(n[0])  # arbitrary height, just for plotting

        # normed_distances_control = control_distances - control_distances.median()


    xlabel('sqrt(distance)')
    ylim()
    format_axis_date(40)
    nicefy()



def plot_duration(cur_result, control_df):
    C = get_subj_control_colors()

    if control_df is not None:
        control_durations = get_control_durations(control_df['processed'])
        cur_dur_flat = (get_duration_per_stim(cur_result) - control_durations.median()).values[1:]
        control_dur_flat = (control_durations - control_durations.median()).iloc[:, 1:].values.flatten()

        _, n, _ = nhist([cur_dur_flat, control_dur_flat], legend=['subject', 'control'], noerror=True, same_bins_flag=True, f=5,
                  maxx=2, color=C, minx=0)
        x = 0
        y = 1.2 * max(n[0])  # arbitrary height, just for plotting
        plot_pin(x, y, color='gray')
        format_axis_date()

    else:
        cur_dur_flat = (get_duration_per_stim(cur_result)).values[1:]
        _, n, _ = nhist(cur_dur_flat, noerror=True, same_bins_flag=True, f=5,
                  maxx=2, color=C)



    xlabel('time (s)')
    nicefy()


def cur_result_to_df(cur_result):
    stimulus = cur_result['stimulus']
    pointer = cur_result['pointer']
    time = cur_result['time']
    new_index = get_new_index(cur_result['selected index'])


    df = pd.DataFrame({'new index': new_index, 'selected index': cur_result['selected index'], 'stimulus': stimulus, 'pointer': pointer,
                         'time': time})

    # replace by nans any pointer positions too far out
    df['pointer'] = df['pointer'].apply(lambda w: np.nan if np.abs(w) > 500 else w)
    # set the stimulus to be the stimulus being targeted (todo: change from 'new stim' to 'targeted stim')

    index_to_stimulus = etl.dictify_cols(df[['selected index', 'stimulus']])
    df['new stim'] = df['new index'].apply(lambda w: index_to_stimulus[w])
    return df


def plot_laser(cur_result):
    df = cur_result_to_df(cur_result)

    index_to_stimulus = etl.dictify_cols(df[['selected index', 'stimulus']])

    cplot(df['stimulus'], '.', color='white', label="_nolegend_")
    cplot(df['pointer'].values[0], marker='o', color='blue', lw=1, markersize=15)
    cplot(df['pointer'].values[-1], marker='*', color='blue', lw=1, markersize=15)

    AX = gca().axis()

    for idx, cur_row in df.groupby('new index'):

        stim_pos = index_to_stimulus[idx]

        #     plot the stimulus position and the pointer path
        cplot(stim_pos, 'o', color='k', markersize=20, fillstyle='none', markeredgewidth=1)
        cplot(cur_row['pointer'].values, color='blue', lw=1)

        #     prep data for the arrow
        S = len(cur_row)
        #     pick where in the path you want the arrow to appear
        n1 = np.int(3 * S / 5)
        n2 = np.int(4 * S / 5)
        v = cur_row['pointer'].values[[n1, n2]]
        #     note rotate 30 degrees for the native angle of the triangle
        cur_angle = 30 + 180 * np.angle(np.diff(v)) / np.pi

        #     plot the arrow and the line-segment
        cplot(cur_row['pointer'].values[n2], marker=(3, 0, cur_angle), color='black', lw=1, markersize=12,
              markeredgewidth=0)
        cplot(v, color='black', lw=2, markersize=15)

            # if the path was too long - here are some example colors to use
            # cplot(cur_row['pointer'].values[n2], marker=(3, 0, cur_angle), color='red', lw=1, markersize=15,
            #       markeredgewidth=0)
            # cplot(stim_pos, 'o', color='red', markersize=20, fillstyle='none', markeredgewidth=2)
            #
            # cplot(v, color='red', lw=1, markersize=15)
            # cplot(cur_row['pointer'].values, color='red', lw=1)

    cplot(df['pointer'].values[0], marker='o', color='blue', lw=1, markersize=15)
    cplot(df['pointer'].values[-1], marker='*', color='blue', lw=1, markersize=15)

    axis('equal')
    legend(["first", "last ", "target", "laser", "direction"], loc='best')
    # axis(AX)
    # axis('off')
    nicefy()
    print(AX)
    # axis('off')

def plot_trailmaking2(processed_df, visit_exam_id=None):
    plot_trailmaking(processed_df, visit_exam_id)


def plot_trailmaking(processed_df, visit_exam_id=None):


    C = get_subj_control_colors()

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    cur_result = current_df.iloc[0]['processed']

    # get control data
    if control_df is not None:
        control_results = control_df['processed'].values
    else:
        control_results = []

    fig_sizer(14, 10)

    subplotter(4, 4, [0, 1, 2, 3, 4, 5, 6, 7])
    plot_laser(cur_result)

    subplotter(4, 4, [8, 9, 10])

    # the index vs. time chart
    plot_trail_accuracy(cur_result)


    # subplotter(2, 3, 3)
    # # the durations between correct target hits
    # old plot not normalized by the control population
    # durations = np.concatenate([w['search durations'][1:] for w in control_results if w['good data']])
    # # to_plot = {'control': durations, 'subject': cur_result['search durations'][1:]}
    # to_plot = [cur_result['search durations'][1:], durations]
    # # nhist(to_plot, f=3, same_bins_flag=True, normalize='frac', color=['gray', C[0]])
    # nhist(to_plot, legend=['subject', 'controls'], f=3, same_bins_flag=True, normalize='frac', color=C)
    # xlabel('duration (s)')
    # nicefy()

    # subplotter(4, 3, 9)
    subplotter(4, 4, 11)
    # the histgram and stem plot of error count

    errors_control = [w['error count'] for w in control_results if w['good data']]
    if len(errors_control) > 0:
        _, n, _ = nhist(errors_control, f=3, color='gray', noerror=True, std_times=1, minx=0)
        y = 1.2 * max(n[0]) # arbitrary height, just for plotting
    else:
        y = 1

    error_subj = cur_result['error count']
    plot_pin(error_subj, y, C[0])
    # plot([error_subj, error_subj], [0, y], linewidth=3, color=C[0])
    # plot([error_subj], [y], 'o', color=C[0], markersize=15)
    # legend(['control', 'subject'])
    xlabel('error count')
    ylim([0, y * 1.3])
    format_axis_date()
    nicefy()

    subplotter(4, 4, [12,13])
    plot_distance(cur_result, control_df)

    subplotter(4, 4, [14,15])
    plot_duration(cur_result, control_df)


    trailmaking_summary = {"test_trailmaking": 0}

    return trailmaking_summary
    # return trailmaking_summary


def plot_categoryfluency(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    if control_df is not None:
        control_result = control_df['processed']
    else:
        control_result = None

    current_result = current_df.iloc[0]['processed']

    fig_sizer(15, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=6, nrows=4, figure=fig, height_ratios=[1,0.25,0.25,0.25])

    category_result = current_result['data']
    category = category_result['category']

    # PLOT PARAMS
    fs = 15
    offset_h = 0.20
    offset_l = .2
    bar_height = 0.5

    # PLOT RESPONSES
    index = 0
    fig.add_subplot(spec[index, 0:6])
    plot_data = category_result['responses']
    plot_data['is_category'] = category_result['is_category']
    plot_data['category_duplicates'] = category_result['category_duplicates']
    plot_data['is_intrusion'] = category_result['is_intrusion']

    for i,v in plot_data.iterrows():
        pointer = 20 - (i % 20)

        if v.is_category:
            color = '#459F5A' # green
            hatch = '////'
            weight = 'roman'
            if v.category_duplicates:
                color = '#EE3A2C' # red
                hatch = 'xxx'
                weight = 'roman'
        elif v.is_intrusion:
            color = '#D6A324' # yellow
            hatch = '|||'
            weight = 'roman'
        else:
            color = '#BAC7D2' # grey
            hatch = False
            weight = False

        plt.gca().add_patch(plt.Rectangle((v.start_time, pointer), v.duration, bar_height, hatch=hatch, edgecolor=color, fill=False))
        # plt.gca().add_patch(plt.Rectangle((v.time_start_seconds, pointer), v.duration, bar_height, color=color))
        plt.text(v.end_time+ offset_l, pointer + offset_h, v.transcript, ha='left', va='center', fontsize=fs, color=color)

    plt.axis('tight')
    plt.gca().set_yticks([])

    #manually make legend
    a_val = 1
    colors = ['#459F5A', '#EE3A2C', '#D6A324', '#BAC7D2']

    circ1 = mpatches.Patch(edgecolor=colors[0], alpha=a_val, hatch='////', label='correct', fill=False)
    circ2 = mpatches.Patch(edgecolor=colors[1], alpha=a_val, hatch='xxx', label='repeat', fill=False)
    circ3 = mpatches.Patch(edgecolor=colors[2], alpha=a_val, hatch='|||', label='intrusion', fill=False)
    circ4 = mpatches.Patch(edgecolor=colors[3], alpha=a_val, hatch=False, label='other', fill=False)

    plt.gca().legend(handles=[circ1, circ2, circ3, circ4], loc='best')

    plt.title("{} Category Fluency Results".format(category))
    nicefy(fsize=20)

    # plot each of the metrics separately in their own bar chart/histogram
    for key in current_result['metrics']:
        sns.set_style('white')
        if index < 3:
            index += 1
            fig.add_subplot(spec[index, 0:6])
            subj_results = current_result['metrics'][key]
            if control_result is not None:
                control_pop_results = np.array(control_result.apply(lambda w: w['metrics'][key]))
                num_controls = len(control_pop_results)
                sns.distplot(control_pop_results, kde=False, color='#BAC7D2')
                y = max(plt.gca().get_ylim())

                # _, n, _ = nhist(control_pop_results, f=3, color='gray', noerror=True)
                # y = max(n[0])
            else:
                num_controls = 0
                y = 1

            plot([subj_results, subj_results], [0, y], linewidth=3, color='#2F366F')
            plot([subj_results], [y], 'o', color='#2F366F', markersize=15)
            ylim([0, y * 1.1])
            label = key.replace('num', '#')
            plt.title("{}, n={}".format(label, num_controls))
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('frequency')
            plt.gca().set_xlabel(label)
            nicefy(fsize=15)

    return 'png'



def plot_letterfluency(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    s3_path = current_df['s3_folder']

    current_result = current_df.iloc[0]['processed']
    if control_df is not None:
        control_result = control_df['processed']
    else:
        control_result = None

    fig_sizer(12, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[1,0.25,0.25])

    letter_result = current_result['data']
    letter = letter_result['letter']

    # PLOT PARAMS
    fs = 15
    offset_h = 0.20
    offset_l = .2
    bar_height = 0.5

    # PLOT RESPONSES
    index=0
    fig.add_subplot(spec[index, 0:6])
    plot_data = letter_result['responses']
    plot_data['correct_letter'] = letter_result['correct_letter']
    plot_data['duplicate_words'] = letter_result['duplicate_word']

    for i, v in plot_data.iterrows():
        pointer = 20 - (i % 20)
        if v.correct_letter:
            color = '#459F5A' # green
            hatch = '////'
            weight = 'roman'
            if v.duplicate_words:
                color = '#EE3A2C' # red
                hatch = 'xxx'
                weight = 'roman'
        else:
            color = '#BAC7D2' # grey
            hatch = False
            weight = False

        plt.gca().add_patch(plt.Rectangle((v.start_time, pointer), v.duration, bar_height, hatch=hatch, edgecolor=color, fill=False))
        plt.text(v.end_time + offset_l, pointer + offset_h, v.transcript, ha='left', va='center',
                 fontsize=fs, color=color, weight=weight)

        plt.axis('tight')
        plt.gca().set_yticks([])

        # make legend manually
        a_val = 1
        colors = ['#459F5A', '#EE3A2C', '#BAC7D2']

        circ1 = mpatches.Patch(edgecolor=colors[0], alpha=a_val, hatch='////', label='correct', fill=False)
        circ2 = mpatches.Patch(edgecolor=colors[1], alpha=a_val, hatch='xxx', label='repeat', fill=False)
        circ3 = mpatches.Patch(edgecolor=colors[2], alpha=a_val, hatch=False, label='other', fill=False)

        plt.gca().legend(handles=[circ1, circ2, circ3], loc='best')
        plt.title("'{}' Letter Fluency Results".format(letter_result['letter']))
        nicefy(fsize=15)

    # plot each metric separately in bar chart/histogram
    for key in current_result['metrics']:
        if index < 2:
            index += 1
            fig.add_subplot(spec[index, 0:6])
            subj_results = current_result['metrics'][key]
            if control_result is not None:
                control_pop_results = np.array(control_result.apply(lambda w: w['metrics'][key]))
                num_controls = len(control_pop_results)
                sns.distplot(control_pop_results, kde=False, color='#BAC7D2')
                y = max(plt.gca().get_ylim())
                # _, n, _ = nhist(control_pop_results, f=3, color='gray', noerror=True)
                # y = max(n[0])
            else:
                num_controls = 0
                y = 1

            plot([subj_results, subj_results], [0, y], linewidth=3, color='#2F366F')
            plot([subj_results], [y], 'o', color='#2F366F', markersize=15)
            ylim([0, y * 1.1])
            label = key.replace('num', '#')
            plt.title("{}, n={}".format(label, num_controls))
            plt.gca().set_yticks([])
            plt.gca().set_ylabel('frequency')
            plt.gca().set_xlabel(label)

            nicefy(fsize=15)

    return 'png'
