from chart.c_general import *


def plot_stroop_timeline(results):
    stim_df = results['stim_df']
    stim_df_sub = results['stim_df_sub']
    transcript = results['transcript']
    stim_df['Total Time'] = stim_df['Total Time'] - stim_df.iloc[0]['Total Time']

    # shift the speech by the time at the start of the exam.
    # once one big audio file is used - we won't need this shift anymore
    # a, shiftt = logfit(results['results_df']['sync_locations'], results['results_df']['total_time'])


    map_color = {'red': 'lightcoral',
                  'purple': 'orchid',
                  'black': 'darkslategrey',
                  'blue': 'cornflowerblue',
                  'green': 'mediumseagreen',
                  'orange': 'sandybrown',
                  'yellow': 'gold',
                  'k': 'silver'}


    #     recalculate start and ends for voice etc.
    stim_start_ends = etl.start_and_ends(stim_df['Active Text'].apply(lambda w: str(w) != 'nan'))
    # stimulus
    ts = stim_df['Total Time'].values

    params = {'fontweight': 'bold', 'rotation': -25, 'rotation_mode': 'anchor'}

    stim_text = .3

    # plot the stimulus
    # TODO: ALIGN THIS PROPERLY, IT'S OFF BY A SECOND OR TWO
    for idx, cur_row in stim_df_sub.iterrows():

        cur_c = map_color[cur_row['Active Color'].lower()]
        plt.text(cur_row['Total Time'], stim_text, cur_row['Active Text'], color=cur_c, **params)
        plot(cur_row['Total Time'], stim_text + .1, '.', color=map_color[cur_row['Active Text'].lower()], markeredgecolor='none', markersize=25)

        # print(stim_start_ends[idx])

        x = [ts[w] for w in stim_start_ends[idx]]
        # plot(x, [.5, .5], color=cur_c, lw=10)
        plt.gca().add_patch(plt.Rectangle((x[0], .4), x[1] - x[0], .1, color=cur_c))

    params = {'rotation': 40, 'rotation_mode': 'anchor'}

    # plot the speech
    for idx, cur_row in transcript.iterrows():
        x = cur_row['start_time'] # + shiftt
        y = .7

        cur_c = cur_row['transcript'].lower()
        if cur_c not in map_color:
            cur_c = 'k'
            plt.text(x, y, cur_row['transcript'], color='k', **params)

        cur_c = map_color[cur_c]
        plt.gca().add_patch(plt.Rectangle((x, .6), cur_row['duration'], .2, color=cur_c))

    xlabel("Time (s)")
    ylim([.1, 1.0])
    xlim([ts[0], ts[-1] + 3])
    yticks([.4, .7])
    gca().set_yticklabels(['stimulus', 'speech'])
    nicefy()


def plot_stroop(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    cur_result = current_df.iloc[0]['processed']

    # get control data
    if control_df is not None:
        control_results = control_df['processed'].values
        control_acc = [w['accuracy'] for w in control_results if 'accuracy' in w]
        control_spd = [w['speed_median'] for w in control_results if 'speed_median' in w]
    else:
        control_acc = []
        control_spd = []

    # ge the dictionary of congruent and incongruent reaction times
    results_df = cur_result['results_df']
    results_correct = results_df[results_df['correct_or_no']]
    reaction_times = etl.dictify_cols2(results_correct[['congruent_or_no', 'reaction_time']])
    cong_mapper = {True: 'congruent', False: 'incongruent'}
    reaction_times = {cong_mapper[k]: v for k, v in reaction_times.items()}

    C = get_subj_control_colors()

    fig_sizer(8,10)
    subplotter(2, 3, [0, 1, 2])
    plot_stroop_timeline(cur_result)

    subplotter(2, 3, 3)
    _ = nhist(reaction_times, f=2, same_bins_flag=True, normalize='frac')
    # legend(loc='best')
    xlabel('reaction time')
    ylabel('% of stimuli')
    ylim([0, 80])

    nicefy(expand_y=True)


    subplotter(2, 3, 4)
    if len(control_acc) > 0:
        _, n, _ = nhist(control_acc, f=3, color='gray', noerror=True)
        y = 1.2 * max(n[0]) # arbitrary height, just for plotting
    else:
        y = 1

    plot_pin(cur_result['accuracy'], y, C[0])
    # legend(['control', 'subject'])
    xlabel('accuracy')
    format_axis_date()
    ylim([0, y * 1.1])
    nicefy()

    subplotter(2, 3, 5)
    if len(control_spd) > 0:
        _, n, _ = nhist(control_spd, f=3, color='gray', noerror=True)
        y = 1.2 * max(n[0]) # arbitrary height, just for plotting
    else:
        y = 1

    plot_pin(cur_result['speed_median'], y, C[0])
    if len(control_spd) > 0:
        legend(['control', 'subject'])
    xlabel('median stimuli per second')
    format_axis_date()
    ylim([0, y * 1.1])
    nicefy()



def plot_bostonnaming_timeline(results):
    # using the results_df for basically everything
    stim_df = results['stim_df']
    transcript = results['transcript']
    results_df = results['results_df']

    # recalculate start and ends for voice etc.
    stim_start_ends = etl.start_and_ends(stim_df['Active Image'].apply(lambda w: str(w) != 'nan'))
    # stimulus
    ts = stim_df['Total Time'].values

    # set attributes for stimulus and speech plotting
    stim_params = {'fontweight': 'bold', 'rotation': -25, 'rotation_mode': 'anchor'}
    speech_params = {'rotation': 40, 'rotation_mode': 'anchor'}

    stim_text = .3
    speech_text = 0.7

    # plot the stimulus below and the answer above
    for idx, cur_row in results_df.iterrows():
        stim_c = 'grey'
        plt.text(cur_row['total_time'], stim_text, cur_row['active_image'], color=stim_c, **stim_params)

        stim_x = [ts[w] for w in stim_start_ends[idx]]
        # plot(x, [.5, .5], color=cur_c, lw=10)
        plt.gca().add_patch(plt.Rectangle((stim_x[0], .4), stim_x[1] - stim_x[0], .1, color=stim_c))

        speech_c = 'green' if cur_row['correct_or_no'] else 'grey'
        if cur_row['reaction_time'] is not None:
            speech_x = cur_row['total_time'] + cur_row['reaction_time']
        else:
            speech_x = cur_row['total_time']
        plt.text(speech_x, speech_text, cur_row['answer'], color=speech_c, **speech_params)
        plt.gca().add_patch(plt.Rectangle((speech_x, .58), cur_row['duration'], .1, color=speech_c))

    xlabel("Time (s)")
    ylim([.1, 1.0])
    xlim([ts[0], ts[-1] + 3])
    yticks([.4, .7])
    gca().set_yticklabels(['stimulus', 'answer'])
    nicefy()
    return


def plot_bostonnaming(processed_df, visit_exam_id=None):
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    cur_result = current_df.iloc[0]['processed']

    # get control data
    if control_df is not None:
        control_results = control_df['processed'].values
        control_acc = [w['accuracy'] for w in control_results if 'accuracy' in w]
        control_spd = [w['speed_median'] for w in control_results if 'speed_median' in w]
    else:
        control_acc = []
        control_spd = []

    # ge the dictionary of congruent and incongruent reaction times
    results_df = cur_result['results_df']
    results_correct = results_df[results_df['correct_or_no']]
    reaction_times = results_df['reaction_time']

    C = get_subj_control_colors()

    fig_sizer(8, 10)
    subplotter(2, 3, [0, 1, 2])
    plot_bostonnaming_timeline(cur_result)

    subplotter(2, 3, 3)
    _ = nhist(reaction_times, f=2, same_bins_flag=True, normalize='frac')
    # legend(loc='best')
    xlabel('reaction time')
    ylabel('% of stimuli')
    ylim([0, 80])

    nicefy(expand_y=True)

    subplotter(2, 3, 4)
    if len(control_acc) > 0:
        _, n, _ = nhist(control_acc, f=3, color='gray', noerror=True)
        y = 1.2 * max(n[0])  # arbitrary height, just for plotting
    else:
        y = 1

    plot_pin(cur_result['accuracy'], y, C[0])
    # legend(['control', 'subject'])
    xlabel('accuracy')
    format_axis_date()
    ylim([0, y * 1.1])
    nicefy()

    subplotter(2, 3, 5)
    if len(control_spd) > 0:
        _, n, _ = nhist(control_spd, f=3, color='gray', noerror=True)
        y = 1.2 * max(n[0])  # arbitrary height, just for plotting
    else:
        y = 1

    plot_pin(cur_result['speed_median'], y, C[0])
    if len(control_spd) > 0:
        legend(['control', 'subject'])
    xlabel('median stimuli per second')
    format_axis_date()
    ylim([0, y * 1.1])
    nicefy()




def plot_digitspanforward(processed_df, visit_exam_id):
    return plot_digitspan(processed_df, visit_exam_id)


def plot_digitspanbackward(processed_df, visit_exam_id):
    return plot_digitspan(processed_df, visit_exam_id)


def plot_digitspan(processed_df, visit_exam_id):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    result = current_df.iloc[0]['processed']

    controls_exist = control_df is not None
    subplotter(2, 1, 0)
    transcript = result['transcript']
    stim_times_end = result['stim_times_end']

    # needed for some reason for text to work, and to set the x axis bounds nicely
    plot([np.min(stim_times_end) - 3, np.max(stim_times_end)], [0, 0], 'white')

    # PLOT PARAMS
    fs = 15
    offset_h = 0.20
    offset_l = -1.2
    bar_height = 0.5
    pointer = 1

    y = .7

    # plot the speech and stimulus
    cnt = 0
    for idx, cur_row in transcript.iterrows():

        #     increment the count if we aren't at the end, AND our text is after the last stimulus
        if cnt < len(result['stim_times_end']) - 1 and cur_row['start_time'] > result['stim_times_start'][cnt + 1]:
            cnt += 1

        x = cur_row['start_time']
        cur_transcript = cur_row['transcript']
        if cur_transcript.isdigit():

            if cur_row['start_time'] < result['stim_times_end'][cnt]:
                cur_c = linspecer(3)[0]
                cur_hatch = ''

            elif result['lev_dist_list'][cnt] == 0:
                cur_c = 'g'
                cur_hatch = '////'

            else:
                cur_c = 'r'
                cur_hatch = 'xxxx'



        else:  # text was spoken
            cur_c = 'gray'
            cur_hatch = False
        plt.gca().add_patch(
            plt.Rectangle((x, y), cur_row['duration'], bar_height, edgecolor=cur_c, fill=False, hatch=cur_hatch))

        plt.text(cur_row['end_time'] + offset_l, pointer + offset_h, cur_transcript, ha='left', va='bottom',
                 fontsize=fs, color=cur_c, rotation=45)

    # plot the stimulus
    # for t, digits in zip(result['stim_times_end'], result['stim_digits']):
    for ii in range(len(result['stim_times_end'])):
        t_start = result['stim_times_start'][ii]#in zip(result['stim_times_end'], result['stim_digits']):
        t_end = result['stim_times_end'][ii]
        digits = result['stim_digits'][ii]
        # plot(t_end, .5, 'k*')
        plt.gca().add_patch(
            plt.Rectangle((t_end - 5, 0.2), 5, .4, edgecolor='none', fill=True, hatch='', facecolor=(linspecer(3)[2] +1 )/2)
        )
        plt.gca().add_patch(
            plt.Rectangle((t_start, 0.2), t_end - t_start, .4, edgecolor='k', fill=False, hatch='')
        )
        plt.text(t_start + .5, .5, digits, ha='left', va='top', fontsize=fs, color='k', rotation=0)

    ylim([0, 2])
    xlabel('time')
    nicefy()

    yticks([.4, 1])
    gca().set_yticklabels(['stimulus', 'speech'])

    subplotter(2, 1, 1)
    C = get_subj_control_colors()
    max_level_subject = result['metrics']['max_level_perfect']

    if control_df is not None:
        max_level_controls = [w['metrics']['max_level_perfect'] for w in control_df['processed'].values]
        _, n, _ = nhist(max_level_controls, f=3, color='gray', noerror=True, std_times=1, minx=0, normalize='frac',
                        int_bins_flag=True)
        ylabel('% of controls')
        y = 1.2 * max(n[0])  # arbitrary height, just for plotting
    else:
        y = 1.2

    ylim([0, y * 1.2])
    plot_pin(max_level_subject, y, C[0])
    xlabel('Highest number of digits correctly recalled')
    nicefy()

