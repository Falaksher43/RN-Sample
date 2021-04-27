from analysis.a_general import *

def process_tapping(time_df, audio_df):
    """
    :param time_df: timeseries used for analysis of left and right triggers
    :param audio_df: unused at this time
    :return:
    """
    # normalize trigger values from 0-1
    time_df['Right Trigger'] = time_df['Right Trigger'] / 255
    time_df['Left Trigger'] = time_df['Left Trigger'] / 255

    # get start of exam (after initialization and tutorial)
    start_idx, _ = utils_df.get_exam_start(time_df)
    time_df = time_df.iloc[start_idx:]

    # find the indices where there was a local maximum for each trigger value on each side
    right_presses = signal.find_peaks(time_df['Right Trigger'], height=0.9)[0] + np.min(time_df.index)
    left_presses = signal.find_peaks(time_df['Left Trigger'], height=0.9)[0] + np.min(time_df.index)

    # label each of those indices in the time_df based on whether they were pressed on the right or the left
    time_df.loc[time_df.index.isin(right_presses), 'side_pressed'] = 1
    time_df.loc[time_df.index.isin(left_presses), 'side_pressed'] = -1

    # just keep the trigger columns for calculations and also filter out the ones that correspond to a max button press
    triggers = time_df.loc[~pd.isnull(time_df['side_pressed']), ['Total Time', 'Right Trigger', 'Left Trigger', 'side_pressed', 'Stimulus']]

    section_dict = dict()
    exam_sections = pd.unique(time_df['Stimulus'])
    TOO_CLOSE_THRESH = 1/12 # seconds

    # calculate different results for each section of the exam (Right, Left, Alternate)
    for section in exam_sections:
        section_dict[section] = dict()
        df = triggers.loc[time_df['Stimulus'] == section]
        unfiltered_df = time_df.loc[time_df['Stimulus'] == section]

        duration = df.iloc[-1]['Total Time'] - df.iloc[0]['Total Time']

        # separate right and left
        right = df.loc[df['side_pressed'] == 1]
        left = df.loc[df['side_pressed'] == -1]

        section_dict[section]['num_right'] = len(right)
        section_dict[section]['num_left'] = len(left)

        # calculate difference in time between each successive press
        # use this to calculate Root Mean Square of Successive Differences (RMSSD)
        df.loc[:, 'diff'] = df['Total Time'].diff(periods=-1)
        section_dict[section]['RMSSD'] = np.sqrt(np.mean(df['diff'] ** 2))

        section_dict[section]['avg_right_value'] = np.mean(unfiltered_df['Right Trigger'])
        section_dict[section]['avg_left_value'] = np.mean(unfiltered_df['Left Trigger'])

        df.loc[:, 'too_close'] = False

        # identify successive trigger presses that occurred below some time threshold that we consider to be simultaneous
        # only applies for alternating task
        if section == 'Alternate':
            df.loc[:, 'too_close'] = abs(df['diff']) < TOO_CLOSE_THRESH
            df.loc[:, 'too_close_temp'] = df['too_close'].shift() == True
            df.loc[:, 'too_close'] = df['too_close'] | df['too_close_temp']
            df = df.drop(['too_close_temp'], axis=1)

            # take the diff of the side_pressed values (-1 and 1)
            # if alternating, should be either 2 or -2. Values of 0 indicate a repeated side (e.g. Right Right Left)
            df['order'] = df['side_pressed'].diff(periods=-1)
            ordering_error_count = len(df.loc[df['order'] == 0])

            # record number of simultaneous presses and number of times that trigger presses weren't alternating
            section_dict[section]['simultaneous_presses'] = int(df['too_close'].sum()/2)
            section_dict[section]['ordering_errors'] = ordering_error_count

    results = dict()
    results['data_by_section'] = section_dict
    results['trigger_data'] = triggers

    metrics = dict()
    metrics['right_section_right_presses'] = section_dict['Right']['num_right']
    metrics['right_section_left_presses'] = section_dict['Right']['num_left']
    metrics['left_section_right_presses'] = section_dict['Left']['num_right']
    metrics['left_section_left_presses'] = section_dict['Left']['num_left']
    metrics['alternate_section_right_presses'] = section_dict['Alternate']['num_right']
    metrics['alternate_section_left_presses'] = section_dict['Alternate']['num_left']
    metrics['ordering_errors'] = section_dict['Alternate']['ordering_errors']
    metrics['next_exam_params'] = None


    return {'data': results,
            'metrics': metrics,
            'has_error': False}


def process_memoryencoding(time_df, audio_dict):
    def get_sub_transcript(transcript, start_ts, end_ts):
        """
        Just get out the things people said between two times
        """
        sub_df = transcript[transcript['start_time'] > start_ts]
        sub_df2 = sub_df[sub_df['end_time'] <= end_ts]
        return sub_df2['transcript'].values

    start_idx, _ = utils_df.get_exam_start(time_df)
    t = time_df['Total Time'].iloc[start_idx:].values
    time_df = time_df.iloc[start_idx:]

    presented_words = pd.unique(time_df['Stimulus'])
    presented_words = list(presented_words)

    # remove NaN and REPEAT from list of presented words
    presented_words.remove(np.nan)
    presented_words.remove('REPEAT')
    num_stimulus_words = len(presented_words)

    # find where the stimulus changes to REPEAT (where the user should respond) and when that period ends
    time_df['change_to_repeat'] = time_df['Stimulus'].apply(lambda x: 1 if x == 'REPEAT' else 0)
    response_periods = etl.start_and_ends(time_df['change_to_repeat'])

    transcript = audio_dict['transcript']

    # make all words lowercase
    transcript['transcript'] = transcript['transcript'].str.lower()

    # convert the times to floats
    transcript[["start_time", "end_time"]] = transcript[["start_time", "end_time"]].apply(pd.to_numeric)

    # calculate duration
    transcript['duration'] = transcript['end_time'] - transcript['start_time']

    performance_dict = dict()
    for idx, time_period in enumerate(response_periods):
        performance_dict[time_period] = dict()

        time_subset = time_df.iloc[time_period[0]:time_period[1]]['Total Time']
        start_time = time_subset.iloc[0]
        end_time = time_subset.iloc[-1]
        response = get_sub_transcript(transcript, time_subset.iloc[0], time_subset.iloc[-1])

        # first identify how many unique words they said and then how many of those were correct
        unique_words = set(response)
        correct_words = unique_words & set(presented_words)
        num_correct = len(correct_words)

        # identify the nouns said during the response period and then identify the ones that were not correct words
        with open('config/nouns.txt') as nounfile:
            list_of_nouns = nounfile.read().splitlines()
        nouns_said = list(unique_words & set(list_of_nouns))
        intrusions = set(nouns_said) - set(correct_words)
        num_intrusions = len(intrusions)

        # we include the -2 bc the presented words also include REPEAT and NaN
        proportion_correct = num_correct / num_stimulus_words

        performance_dict[time_period]['trial'] = idx + 1
        performance_dict[time_period]['start_time'] = start_time
        performance_dict[time_period]['end_time'] = end_time
        performance_dict[time_period]['num_correct'] = num_correct
        performance_dict[time_period]['proportion_correct'] = proportion_correct
        performance_dict[time_period]['intrusions'] = intrusions
        performance_dict[time_period]['num_intrusions'] = num_intrusions

    results_df = pd.DataFrame.from_dict(performance_dict).transpose()

    max_words_correct = np.max(results_df['num_correct'])
    total_intrusions = np.sum(results_df['num_intrusions'])

    NUM_WORDS_MIN = 5
    NUM_WORDS_MAX = 15

    # choose stimuli going up or down:
    stim_words_bump = (np.max(results_df['proportion_correct']) > 0.7) - (np.max(results_df['proportion_correct']) < 0.5)

    # limit bounds
    next_exam_param = np.max([NUM_WORDS_MIN, np.min([NUM_WORDS_MAX, num_stimulus_words + stim_words_bump*5])])

    results = dict()
    results['transcript'] = transcript
    results['response_periods'] = response_periods
    results['presented_words'] = presented_words
    results['performance_by_section'] = results_df

    metrics = dict()
    metrics['max_words_correct'] = max_words_correct
    metrics['num_intrusions'] = total_intrusions
    metrics['next_exam_params'] = next_exam_param

    return {'data': results,
            'metrics': metrics,
            'has_error': False}


def process_memoryrecall(time_df, audio_dict):

    # determine which words they were shown in the associated encoding task
    set_names = ['Set 1', 'Set 2', 'Set 3']
    encoding_record = list(pd.unique(time_df['Stimulus']))
    encoding_record.remove(np.nan)
    encoding_words = set(encoding_record) - set(set_names)
    num_encoding_words = len(encoding_words)

    # have to account for the initialization, which could vary. Audio recorded from "Tutorial" onwards
    very_beginning = time_df.loc[time_df['Section'] == 'Recall']
    recall_tutorial = very_beginning.loc[very_beginning['Exam Status'] == 'Tutorial']
    recall_tutorial_time = recall_tutorial.iloc[0]['Total Time']

    start_idx, _ = utils_df.get_exam_start(time_df)
    t = time_df['Total Time'].iloc[start_idx:].values
    time_df = time_df.iloc[start_idx:]

    # subtract the initialization/VR experience, allows us to align to the audio file
    time_df['Total Time'] = time_df['Total Time'] - recall_tutorial_time

    # identify the time periods for the two parts of the exam
    recall_period = time_df.loc[time_df['Section'] == 'Recall']
    recall_start_time = recall_period.iloc[0]['Total Time']

    recognize_period = time_df.loc[time_df['Section'] == 'Familiarity']
    recognize_start_time = recognize_period.iloc[0]['Total Time']

    transcript = audio_dict['transcript']

    # make all words lowercase
    transcript['transcript'] = transcript['transcript'].str.lower()

    # convert the times to floats
    transcript[["start_time", "end_time"]] = transcript[["start_time", "end_time"]].apply(pd.to_numeric)

    # calculate duration
    transcript['duration'] = transcript['end_time'] - transcript['start_time']

    # filter out the part of the transcript that corresponds to the recall portion
    recall_transcript = transcript[
        (transcript['start_time'] > recall_start_time) & (transcript['start_time'] < recognize_start_time)]

    # identify words that have been recalled correctly
    recalled_correctly = recall_transcript.transcript.apply(lambda x: x in encoding_words)
    recall_transcript['recalled_correctly'] = recalled_correctly

    # check for words that have been separated ('mud slide') --> ('mudslide')
    for row1, row2 in more_itertools.pairwise(recall_transcript.iterrows()):
        if row1[1]['recalled_correctly'] == False & row2[1]['recalled_correctly'] == False:
            possibly_correct = row1[1]['transcript'] + row2[1]['transcript']
            if possibly_correct in encoding_words:
                recall_transcript.loc[recall_transcript['start_time'] == row1[1]['start_time'], 'recalled_correctly'] = True

                # combine two words to make 1 row --> makes for better plotting and counting
                recall_transcript.loc[recall_transcript['start_time'] == row1[1]['start_time'], 'transcript'] = possibly_correct

                # removing second word of pairwise combo from dataframe
                recall_transcript = recall_transcript.loc[~(recall_transcript['start_time'] == row2[1]['start_time'])]

    # label intrusions (must be a noun to be an intrusion)
    with open('config/nouns.txt') as nounfile:
        list_of_nouns = nounfile.read().splitlines()

    recall_transcript['is_intrusion'] = recall_transcript['transcript'].apply(
        lambda x: (x not in encoding_words) & (x in list_of_nouns))

    # calculate some metrics relating to the recall section
    recall_num_correct = np.sum(recall_transcript['recalled_correctly'])
    recall_num_intrusions = np.sum(recall_transcript['is_intrusion'])

    # filter out the part of the transcript that corresponds to the recognize portion
    recognize_transcript = transcript[transcript['start_time'] > recognize_start_time]

    key_columns = ['Total Time', 'Stimulus']

    stim_df = recognize_period[key_columns]

    # get the start and end times for stimuli being displayed
    stim_start_ends = etl.start_and_ends(stim_df['Stimulus'].apply(lambda w: str(w) != 'nan'))

    # df with one row per stimulus
    stim_df_sub = stim_df.iloc[np.array(stim_start_ends)[:, 0] + 1, :]
    stim_df_sub = stim_df_sub.reset_index()

    results_df = pd.DataFrame(columns=['total_time', 'stimulus', 'answer',
                                       'correct_or_no'])

    # todo: add more affirmative and negative words once we have more data on what people say
    affirmative_words = ['yes', 'yeah']
    negative_words = ['no', 'nope']
    for i, cur_row in stim_df_sub.iterrows():

        answers_dict = dict()
        # identify words that fell after the first chirp
        possible_words = transcript.loc[(transcript['start_time'] > stim_df_sub.iloc[i]['Total Time'])]

        # if not the last stim, select all words that occur before next chirp
        if i < len(stim_df_sub) - 1:
            possible_words = possible_words.loc[(transcript['start_time'] < stim_df_sub.iloc[i + 1]['Total Time'])]

        # if words were said during this time
        if len(possible_words) > 0:

            # take the first thing they said
            answer = possible_words.iloc[0]['transcript']

            # determine if answer was correct or not and add to correct results dict
            if (answer in affirmative_words) and (cur_row['Stimulus'] in encoding_words):
                correct = True
            elif (answer in negative_words) and (cur_row['Stimulus'] not in encoding_words):
                correct = True
            else:
                correct = False

        else:
            answer = 'blank'
            correct = False

        answers_dict['total_time'] = cur_row['Total Time']
        answers_dict['stimulus'] = cur_row['Stimulus']
        answers_dict['answer'] = answer
        answers_dict['correct_or_no'] = correct

        results_df = results_df.append(answers_dict, ignore_index=True)

    recognize_without_tutorial = results_df.iloc[3:]
    recognize_num_correct = recognize_without_tutorial['correct_or_no'].sum()

    # record results and metrics
    results = dict()
    results['encoding_words'] = encoding_words
    results['recall_transcript'] = recall_transcript  # contains subject's answers + timestamps
    results['recognize_results'] = results_df  # contains tutorial + exam sections, presented stimulus, subject's answers, correct or not

    metrics = dict()
    metrics['recall_num_correct'] = recall_num_correct
    metrics['recall_num_intrusions'] = recall_num_intrusions
    metrics['recognize_num_correct'] = recognize_num_correct
    metrics['next_exam_params'] = None

    return {'data': results,
            'metrics': metrics,
            'has_error': False}

