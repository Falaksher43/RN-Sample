from chart.c_general import *

def plot_tapping(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    if control_df is not None:
        control_result = control_df['processed']
    else:
        control_result = None

    current_result = current_df.iloc[0]['processed']

    fig_sizer(15, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=6, nrows=8, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
    index = 0
    metrics_to_viz = ['right_section_right_presses', 'left_section_left_presses', 'ordering_errors']
    title_dict = {'right_section_right_presses': 'Right Section Right Presses',
                  'left_section_left_presses': 'Left Section Left Presses',
                  'ordering_errors': 'Alternate Section Ordering Errors'}

    # plot each of the metrics separately in their own bar chart/histogram
    for index, key in enumerate(metrics_to_viz):
        sns.set_style('white')
        fig.add_subplot(spec[index, 0:6])
        subj_results = current_result['metrics'][key]
        if control_result is not None:
            control_pop_results = np.array(control_result.apply(lambda w: w['metrics'][key]))
            num_controls = len(control_pop_results)
            sns.distplot(control_pop_results, kde=False, color='#BAC7D2')
            y = max(plt.gca().get_ylim())

        else:
            num_controls = 0
            y = 1

        plot([subj_results, subj_results], [0, y], linewidth=3, color='#2F366F')
        plot([subj_results], [y], 'o', color='#2F366F', markersize=15)
        ylim([0, y * 1.1])
        label = title_dict[key]
        plt.title("{}, n={}".format(label, num_controls))
        plt.gca().set_yticks([])
        plt.gca().set_ylabel('frequency')
        nicefy(fsize=15)

    return 'png'


def plot_memoryencoding(processed_df, visit_exam_id=None):
    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    current_result = current_df.iloc[0]['processed']
    presented_words = current_result['data']['presented_words']
    print(presented_words)

    if control_df is not None:
        control_result = control_df['processed']
    else:
        control_result = None

    fig_sizer(15, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[1, 1, 1])

    metrics_to_viz = ['max_words_correct', 'num_intrusions']
    title_dict = {'max_words_correct': 'Encoding: Maximum Words Correct',
                  'num_intrusions': 'Encoding: Number of Intrusions'}
    # plot each of the metrics separately in their own bar chart/histogram
    for index, key in enumerate(metrics_to_viz):
        sns.set_style('white')
        fig.add_subplot(spec[index, 0:6])
        subj_results = current_result['metrics'][key]
        if control_result is not None:
            control_pop_results = np.array(control_result.apply(lambda w: w['metrics'][key]))
            num_controls = len(control_pop_results)
            sns.distplot(control_pop_results, kde=False, color='#BAC7D2')
            y = max(plt.gca().get_ylim())

        else:
            num_controls = 0
            y = 1

        plot([subj_results, subj_results], [0, y], linewidth=3, color='#2F366F')
        plot([subj_results], [y], 'o', color='#2F366F', markersize=15)
        ylim([0, y * 1.1])
        xlim([0, len(presented_words)]) # -2 because presented words still has NaN and REPEAT
        label = title_dict[key]
        plt.title("{}, n={}".format(label, num_controls))
        plt.gca().set_yticks([])
        plt.gca().set_ylabel('frequency')
        nicefy(fsize=15)

    return 'png'

def plot_memoryrecall(processed_df, visit_exam_id=None):

    current_df, control_df = utils_db.split_current_control_df(processed_df, visit_exam_id)
    current_result = current_df.iloc[0]['processed']

    if control_df is not None:
        control_result = control_df['processed']
    else:
        control_result = None

    fig_sizer(15, 12)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[1,1,1])


    metrics_to_viz = ['recall_num_correct', 'recall_num_intrusions', 'recognize_num_correct']
    title_dict = {'recall_num_correct': 'Recall: Number Correct',
                  'recall_num_intrusions': 'Recall: Number of Intrusions',
                  'recognize_num_correct': 'Recognize: Number Correct'}
    # plot each of the metrics separately in their own bar chart/histogram
    for index, key in enumerate(metrics_to_viz):
        sns.set_style('white')
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
        xlim([0,10])
        label = title_dict[key]
        plt.title("{}, n={}".format(label, num_controls))
        plt.gca().set_yticks([])
        plt.gca().set_ylabel('frequency')
        nicefy(fsize=15)

    return 'png'