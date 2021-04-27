from analysis.a_general import *



def process_stroop(time_df, audio_dict):
    def process_stopped_talking(audio_dict):
        '''
            Simulating speech detection implemented in Unity Code.

            Note: The wait time between detecting when a speaker stops talking
            and the restart of the detector assumes 0.75 second wait. The actual
            implementation had a random wait time between 0.75 and 1.25 seconds.

            :return: timestamps for when speaker stopped talking
        '''

        # LOADING DATA
        # filepath = audio_dict['stroop_filepath']
        # data, sample_rate = audio_dict['stroop_librosa']
        audio = audio_dict['stroop']
        sample_rate = audio.frame_rate

        data = convert_to_librosa(audio, sample_rate)

        # DEFINING MINIMUM THRESHOLD IN WHICH SPEECH IS DETECTED
        # Value from Unity Code.
        minimumLevel = 10

        # NUMBER OF SAMPLES OVER WHICH
        # Value from Unity Code.
        sampleCount = 1024

        # INITIALIZING SPEECH TRACKING
        # I presume it was initialized as false (not in the Unity version)
        startedTalking_flag = False
        startedTalking = []

        # I presume it was false during the whole audio
        triggerPressed = False

        # I presume it was initialized as false (not in the Unity version)
        stoppedTalking_flag = False
        stoppedTalking = []

        # INDEX TO CLIP SPEECH SEGMENT
        index = -1

        # LOOPING THROUGH EACH SAMPLE
        for i, _ in enumerate(data):

            index = index + 1

            # STOP PROCESSING IF INDEX IS BEYOND DATA LENGTH
            if index > len(data):
                break

            # CHECKING IF WE HAVE ENOUGH SAMPLES IN DATA TO EVALUATE ENERGY
            if index < sampleCount:
                continue

            # ---------------------------------------------------
            # CALCULATE ENERGY OVER WINDOW
            # ---------------------------------------------------
            clipSamples = data[index - sampleCount:index]

            # CALCULATE VOLUME / ENERGY
            SUM = np.sum(np.abs(clipSamples))

            # ---------------------------------------------------
            # KEEP TRACK OF SPEAKING PHASE
            # ---------------------------------------------------

            # CHECK IF SUBJECT STARTED SPEAKING
            if SUM > minimumLevel:

                startedTalking_flag = True
                # startedTalking.append(index / sample_rate)

            # CHECK IF SUBJECT STOPPED SPEAKING
            elif (startedTalking_flag is True) and (SUM < minimumLevel):

                # TRACKING WHEN SPEAKER STOPPED
                stoppedTalking_flag = True
                stoppedTalking.append(index / sample_rate)

                startedTalking_flag = False

                # ---------------------------------------------------
                # WAITING UNTIL NEXT EVALUATION OF SPEECH
                # ---------------------------------------------------

                # wait for 0.25 seconds + lower threshold of stimulus wait function (0.5s)
                waitTime = 0.25
                waitTime_rand = 0.5
                index = index + int(sample_rate * (waitTime + waitTime_rand))

        return stoppedTalking

    def correct_color(c):
        """
        Fix common color transcription errors
        :param c:
        :return:
        # """
        # todo: use spacey to actually remove punctuation
        # remove any punctuation

        c = c.replace(".", "").replace("?", "").replace(",", "")
        c = c.lower()
        color_dict = {'hello': 'yellow',
                      'yellowish': 'yellow',
                      'yello': 'yellow',
                      'bluish': 'blue',
                      'boo': 'blue',
                      'blew': 'blue',
                      'reddish': 'red',
                      'read': 'red',
                      'reds': 'red',
                      'bed': 'red',
                      'bread': 'red',
                      'fred': 'red',
                      'cred': 'red',
                      'plaque': 'black',
                      'lock': 'black',
                      'glock': 'black',
                      'whack': 'black',
                      'buck': 'black',
                      'block': 'black',
                      'greenish': 'green',
                      'lime': 'green',
                      'ring': 'green',
                      'grain': 'green',
                      'rain': 'green',
                      'queen': 'green',
                      'greed': 'green',
                      'bream': 'green',
                      'greens': 'green',
                      'yellowish': 'yellow',
                      'gold': 'yellow',
                      'triple': 'purple',
                      'purplish': 'purple'}

        return color_dict[c] if c in color_dict else c

    # audiosegment file
    audio = audio_dict['stroop']
    sr = audio.frame_rate

    # get highpassed audio, chirp sound, and bandpassed audio
    hp_audio, hp_sync, bp_audio = process_audio_w_chirp(audio)

    # cross correlation values and sync locations
    corr, ts, sync_locations = get_peak_corr(hp_audio, hp_sync, sr)

    # get all of the idxs where the threshold was passed
    threshold_passed_idxs = threshold_passed(bp_audio, -8)

    # colors to check/we expect to see
    colors = ['red', 'blue', 'orange']

    transcript = audio_dict['transcript']
    transcript = transcript.loc[transcript['transcript'].apply(type) == str]
    transcript['transcript'] = transcript['transcript'].apply(lambda x: x.lower())
    transcript['start_time'] = transcript['start_time'].astype('float')
    transcript['end_time'] = transcript['end_time'].astype('float')
    transcript['duration'] = transcript['end_time'] - transcript['start_time']

    # shiftt = 3.15
    # transcript['start_time'] = transcript['start_time'] + shiftt
    # transcript['end_time'] = transcript['end_time'] + shiftt

    transcript['transcript'] = transcript['transcript'].apply(correct_color)

    stim_df = utils_df.get_df_stroop(time_df)

    start_idx, _ = utils_df.get_exam_start(time_df)
    first_timestamp = time_df.iloc[0]['Total Time']
    ds = np.median(time_df['Delta Time'])

    starts_ends = transcript[['start_time', 'end_time']].values

    # get the start and end times for stimuli being displayed
    stim_start_ends = etl.start_and_ends(stim_df['Active Text'].apply(lambda w: str(w) != 'nan'))

    # df with one row per stimulus
    stim_df_sub = stim_df.iloc[np.array(stim_start_ends)[:, 0] + 1, :]
    stim_df_sub = stim_df_sub.reset_index()

    # need to normalize the stimuli to start at time 0 (this is when the audio file starts recording)
    stim_df_sub['Total Time'] = stim_df_sub['Total Time'] - stim_df_sub.iloc[0]['Total Time']

    # keep track of the time of the threshold passed closest to the amazon transcript color word
    closest_thresh_passed = []

    results_df = pd.DataFrame(columns=['active_color', 'active_text', 'total_time', 'answer',
                                       'correct_or_no', 'congruent_or_no', 'reaction_time',
                                       'reaction_time_amazon', 'fooled'])

    # iterate through every sync location and look between 2 successive ones for values
    for i, cur_row in stim_df_sub.iterrows():

        answers_dict = dict()
        # identify words that fell after the current stimulus
        possible_words = transcript.loc[(transcript['start_time'] > cur_row['Total Time'])]

        # if not the last stim, select all words that occur before next stimulus
        if i < len(stim_df_sub) - 1:
            possible_words = possible_words.loc[(transcript['start_time'] < stim_df_sub.iloc[i + 1]['Total Time'])]

        # check if these words contain a color word
        if pd.Series(colors).isin(possible_words['transcript']).any():
            # get index and start time of first word that is a color
            criteria_met = [meets_criteria(word, colors) for word in possible_words['transcript']]
            first_idx = np.argwhere(criteria_met)[0][0]
            first_timestamp = possible_words.iloc[first_idx]['start_time']
            first_color = possible_words.iloc[first_idx]['transcript']

            # search 0.1 seconds before amazon transcript time for threshold crossing
            search_space = (first_timestamp - 0.1) * 44100
            # get first value closest to search space
            closest_idx = np.abs(threshold_passed_idxs - search_space).argmin()
            closest_thresh = threshold_passed_idxs[closest_idx][0] / 44100
            closest_thresh_passed.append(closest_thresh)

            # calculate how long after chirp the threshold was passed
            reaction_time = closest_thresh - cur_row['Total Time']
            reaction_time_amazon = first_timestamp - cur_row['Total Time']

        else:
            first_color = 'blank'
            reaction_time = None
            reaction_time_amazon = None

        # determine if answer was correct or not and add to correct results dict
        correct = True if cur_row['Active Color'].lower() == first_color else False
        congruent = True if cur_row['Active Text'] == cur_row['Active Color'] else False

        if correct == False and cur_row['Active Text'].lower() == first_color:
            fooled = True
        else:
            fooled = False

        answer = first_color

        answers_dict['active_color'] = cur_row['Active Color']
        answers_dict['active_text'] = cur_row['Active Text']
        answers_dict['total_time'] = cur_row['Total Time']
        answers_dict['correct_or_no'] = correct
        answers_dict['congruent_or_no'] = congruent
        answers_dict['reaction_time'] = reaction_time
        answers_dict['reaction_time_amazon'] = reaction_time_amazon
        answers_dict['answer'] = answer
        answers_dict['fooled'] = fooled

        results_df = results_df.append(answers_dict, ignore_index=True)

    stim_end_times = stim_df.iloc[np.array(stim_start_ends)[:, 1], :]['Total Time'].values
    voice_end_times = np.array(starts_ends)[:, 1]
    results_df = results_df.reset_index()

    # calculate average speed and accuracy
    n_right = len(results_df.loc[results_df['correct_or_no'] == True])
    n_wrong = len(results_df.loc[results_df['correct_or_no'] == False])
    accuracy = n_right / (n_right + n_wrong + etl.eps)

    # astype(float) required until this numpy bug is solved
    # https://github.com/numpy/numpy/issues/10393
    speed_mean = np.nanmean(results_df.loc[results_df['correct_or_no'] == True, 'reaction_time'].values.astype(float))
    speed_median = np.nanmedian(
        results_df.loc[results_df['correct_or_no'] == True, 'reaction_time'].values.astype(float))

    # stopped_talking_timestamps = process_stopped_talking(audio_df)

    return {'transcript': transcript,
            'results_df': results_df,
            'sync_locations': sync_locations,
            'audio_ms': len(audio_dict['stroop']),
            'stim_df': stim_df,
            'stim_df_sub': stim_df_sub,  # one row per stimulus
            'start_idx': start_idx,  # the start of the experiment
            'first_timestamp': first_timestamp,
            'accuracy': accuracy,
            'speed_mean': speed_mean,
            'speed_median': speed_median,
            'stim_end_times': stim_end_times,
            'voice_end_times': voice_end_times,
            # 'stopped_talking_timestamps': stopped_talking_timestamps,
            'ds': ds,

            'metrics': {
                'speed_median': speed_median,
                'accuracy': 100 * accuracy,
                'num_correct': n_right,
                'next_exam_params': None
            },

            'has_error': False}


def process_bostonnaming(time_df, audio_dict):
    def correct_spelling(w):
        """
        Fix common word transcription errors
        :param c:
        :return:
        # """
        # todo: use spacey to actually remove punctuation
        # remove any punctuation

        w = w.replace(".", "").replace("?", "").replace(",", "")
        w = w.lower()
        spelling_dict = {'hangar': 'hanger',
                         'stilt': 'stilts',
                         'rhino': 'rhinoceros',
                         'raina': 'rhinoceros',
                         'racket': 'racquet'}

        return spelling_dict[w] if w in spelling_dict else w

    # audiosegment file
    audio = audio_dict['bostonnaming']
    sr = audio.frame_rate

    # get highpassed audio, chirp sound, and bandpassed audio
    hp_audio, hp_sync, bp_audio = process_audio_w_chirp(audio)

    # cross correlation values and sync locations
    corr, ts, sync_locations = get_peak_corr(hp_audio, hp_sync, sr)

    # a semi-arbitrary number that came from looking at the plot of the waveform in log scale
    # todo: make this more algorithmic once we have more data
    SPEECH_THRESHOLD = -8

    # get all of the idxs where the threshold was passed
    threshold_passed_idxs = threshold_passed(bp_audio, SPEECH_THRESHOLD)

    with open('config/nouns.txt') as nounfile:
        list_of_nouns = nounfile.read().splitlines()

    transcript = audio_dict['transcript']
    transcript = transcript.loc[transcript['transcript'].apply(type) == str]
    transcript['transcript'] = transcript['transcript'].apply(lambda x: x.lower())
    transcript['start_time'] = transcript['start_time'].astype('float')
    transcript['end_time'] = transcript['end_time'].astype('float')
    transcript['duration'] = transcript['end_time'] - transcript['start_time']

    transcript['transcript'] = transcript['transcript'].apply(correct_spelling)

    stim_df = utils_df.get_df_bostonnaming(time_df)
    stim_df['Total Time'] = stim_df['Total Time'] - stim_df['Total Time'].iloc[0]

    start_idx, _ = utils_df.get_exam_start(time_df)
    first_timestamp = time_df.iloc[0]['Total Time']
    ds = np.median(time_df['Delta Time'])

    starts_ends = transcript[['start_time', 'end_time']].values

    # get the start and end times for stimuli being displayed
    stim_start_ends = etl.start_and_ends(stim_df['Active Image'].apply(lambda w: str(w) != 'nan'))

    # df with one row per stimulus
    stim_df_sub = stim_df.iloc[(np.array(stim_start_ends)[:, 0] + 1), :]
    stim_df_sub = stim_df_sub.reset_index()
    stim_df_sub['Total Time'] = stim_df_sub['Total Time'] - stim_df_sub.iloc[0]['Total Time']


    # keep track of the time of the threshold passed closest to the amazon transcript color word
    closest_thresh_passed = []

    results_df = pd.DataFrame(columns=['active_image', 'total_time', 'answer',
                                       'correct_or_no', 'reaction_time',
                                       'reaction_time_amazon'])

    stim_words = np.unique(stim_df_sub['Active Image'].str.lower())

    # iterate through every sync location and look between 2 successive ones for values
    for i, cur_row in stim_df_sub.iterrows():

        answers_dict = dict()
        # identify words that fall after the first stimulus
        possible_words = transcript.loc[(transcript['start_time'] > cur_row['Total Time'])]

        # if not the last stim, select all words that occur before next stimulus
        if i < len(stim_df_sub) - 1:
            possible_words = possible_words.loc[(transcript['start_time'] < stim_df_sub.iloc[i + 1]['Total Time'])]

        # first we check if they said any of the stimulus words
        # if not, we then check to see if they ever said a noun
        if pd.Series(stim_words).isin(possible_words['transcript']).any():
            criteria_met = [meets_criteria(word, stim_words) for word in possible_words['transcript']]
        elif pd.Series(list_of_nouns).isin(possible_words['transcript']).any():
            criteria_met = [meets_criteria(word, list_of_nouns) for word in possible_words['transcript']]
        else:
            criteria_met = [False]

        # if any of the above criteria have been met, identify where it happened
        if any(criteria_met):
            # get index and start time of first word that meets the criteria
            first_idx = np.argwhere(criteria_met)[0][0]
            first_timestamp = possible_words.iloc[first_idx]['start_time']
            duration = possible_words.iloc[first_idx]['duration']
            first_noun = possible_words.iloc[first_idx]['transcript']

            # search 0.1 seconds before amazon transcript time for threshold crossing
            search_space = (first_timestamp - 0.1) * 44100
            # get first value closest to search space
            closest_idx = np.abs(threshold_passed_idxs - search_space).argmin()
            closest_thresh = threshold_passed_idxs[closest_idx][0] / 44100
            closest_thresh_passed.append(closest_thresh)

            # calculate how long after chirp the threshold was passed
            reaction_time = closest_thresh - cur_row['Total Time']
            reaction_time_amazon = first_timestamp - cur_row['Total Time']

        else:
            first_noun = 'blank'
            reaction_time = None
            reaction_time_amazon = None
            duration = 0

        # determine if answer was correct or not and add to correct results dict
        correct = True if cur_row['Active Image'].lower() == first_noun else False

        answers_dict['active_image'] = cur_row['Active Image']
        answers_dict['total_time'] = cur_row['Total Time']
        answers_dict['correct_or_no'] = correct
        answers_dict['reaction_time'] = reaction_time
        answers_dict['reaction_time_amazon'] = reaction_time_amazon
        answers_dict['answer'] = first_noun
        answers_dict['duration'] = duration

        results_df = results_df.append(answers_dict, ignore_index=True)

    stim_end_times = stim_df.iloc[np.array(stim_start_ends)[:, 1], :]['Total Time'].values
    voice_end_times = np.array(starts_ends)[:, 1]
    results_df = results_df.reset_index()

    # calculate average speed and accuracy
    n_right = len(results_df.loc[results_df['correct_or_no'] == True])
    n_wrong = len(results_df.loc[results_df['correct_or_no'] == False])
    accuracy = n_right / (n_right + n_wrong + etl.eps)

    # astype(float) required until this numpy bug is solved
    # https://github.com/numpy/numpy/issues/10393
    speed_mean = np.nanmean(results_df.loc[results_df['correct_or_no'] == True, 'reaction_time'].values.astype(float))
    speed_median = np.nanmedian(
        results_df.loc[results_df['correct_or_no'] == True, 'reaction_time'].values.astype(float))

    return {'transcript': transcript,
            'results_df': results_df,
            'sync_locations': sync_locations,
            'audio_ms': len(audio_dict['bostonnaming']),
            'stim_df': stim_df,
            'stim_df_sub': stim_df_sub,  # one row per stimulus
            'start_idx': start_idx,  # the start of the experiment
            'first_timestamp': first_timestamp,
            'accuracy': accuracy,
            'speed_mean': speed_mean,
            'speed_median': speed_median,
            'stim_end_times': stim_end_times,
            'voice_end_times': voice_end_times,
            # 'stopped_talking_timestamps': stopped_talking_timestamps,
            'ds': ds,

            'metrics': {
                'speed_median': speed_median,
                'accuracy': 100 * accuracy,
                'num_correct': n_right,
                'next_exam_params': None
            },

            'has_error': False}



def process_digitspanforward(time_df, audio_dict):
    return process_digitspan(time_df, audio_dict, min_digits=4, reverse=False)


def process_digitspanbackward(time_df, audio_dict):
    return process_digitspan(time_df, audio_dict, min_digits=3, reverse=True)


def process_digitspan(time_df, audio_dict, min_digits=4, reverse=False):
    def parse_digit_active(t, digit_active):
        """
        For digitspan
        :param selected_index:  -1's and digits
        :return: list of strings where each number is recorded
        """
        # todo: make this work with digit index in case there are repeated numbers

        #   find places where the digit changes
        I = np.abs(np.diff(digit_active)) > 0
        digit_list = digit_active[1:][I]
        digit_str = ''.join([str(w) for w in digit_list])
        digit_str_list = digit_str.split("-1")[:-1]
        #   removing the '-2' at the end. todo: code this better pleasea
        digit_str_list = [w[:-2] for w in digit_str_list]

        # get the time values for the times that the stim presentation *ended
        # todo: look for the digit_active to be == -2 indicating the green bar is in action
        idxs = [w[0] for w in etl.start_and_ends(digit_active != -1)]
        time_list_start = t[idxs].tolist()
        idxs = [w[1] for w in etl.start_and_ends(digit_active != -1)]
        # todo: note: need to remove hard-code in 5 seconds and look for -2
        time_list_end = np.array(t[idxs].tolist())

        if len(time_list_end) != len(digit_str_list):
            raise Exception("Number of digits doesn't match number of timestamps for them")

        return time_list_start, time_list_end, digit_str_list

    def get_sub_transcript(transcript, start_ts, end_ts):
        """
        Just get out the things people said between two times
        """
        sub_df = transcript[transcript['start_time'] > start_ts]
        sub_df2 = sub_df[sub_df['end_time'] <= end_ts]
        return sub_df2['transcript'].values

    numeral_to_digit = {'one': '1',
                        'won': '1',
                        'two': '2',
                        'to': '2',
                        'too': '2',
                        'three': '3',
                        'four': '4',
                        'for': '4',
                        'five': '5',
                        'six': '6',
                        'seven': '7',
                        'eight': '8',
                        'ate': '8',
                        'nine': '9',
                        'zero': '0'}

    start_idx, _ = utils_df.get_exam_start(time_df)

    t = time_df['Total Time'].iloc[start_idx:].values
    digit_active = time_df['Active Digit'].iloc[start_idx:].values
    # digit_index = time_df['Digit Index'].iloc[start_idx:].values

    # mess with the transcript:
    transcript = audio_dict['transcript']
    transcript = transcript.loc[transcript['transcript'].apply(type) == str]

    # convert the times to floats
    transcript[["start_time", "end_time"]] = transcript[["start_time", "end_time"]].apply(pd.to_numeric)
    # todo: WATCH OUT FOR THIS SHIFT BY FIVE HERE!!! 5
    # transcript[["start_time", "end_time"]] = transcript[["start_time", "end_time"]].apply(lambda w: w + 5)
    # shift the times by the tutorial
    # todo: when it is just one audio file remove this
    transcript[["start_time", "end_time"]] = transcript[["start_time", "end_time"]].apply(lambda w: w + t[0])
    # calculate duration
    transcript['duration'] = transcript['end_time'] - transcript['start_time']

    # convert any of those string responses that look like 'seven' into '7'
    convert_nums = lambda w: numeral_to_digit[w.lower()] if w.lower() in numeral_to_digit else w
    transcript['transcript'] = transcript['transcript'].apply(convert_nums)

    stim_times_start, stim_times_end, stim_digits = parse_digit_active(t, digit_active)

    responses = []
    for idx, cur_time in enumerate(stim_times_end):
        #     find the time between this stimulus and the start of the next one, see anything said here
        if idx < len(stim_times_end) - 1:
            end_time = stim_times_start[idx + 1]
        else:
            end_time = np.inf

        #   just the string responses as a list for the time period they were allowed to answer questions for.
        responses_list = get_sub_transcript(transcript, cur_time, end_time).astype(str)

        # remove any non-numeric responses
        numeric_filter = lambda mixed_str: "".join(filter(str.isdigit, mixed_str))
        cur_response = numeric_filter("".join(responses_list))
        responses.append(cur_response)

    if reverse:
        responses_for_eval = [w[::-1] for w in responses]
    else:
        responses_for_eval = responses

    # calculate the levenstein distance
    lev_dist_list = [levenshtein_distance(a, b) for a, b in zip(responses_for_eval, stim_digits)]
    set_dist_list = [set_match(a, b) for a, b in zip(responses_for_eval, stim_digits)]
    dyslexic_swap_list = [dyslexic_swap_match(a, b) for a, b in zip(responses_for_eval, stim_digits)]
    dyslexic_flip_list = [dyslexic_flip_match(a, b) for a, b in zip(responses_for_eval, stim_digits)]

    # dictionary of levenstien distances indexed by number of digits
    num_digits = [len(w) for w in stim_digits]
    dist_dict = {k: v for k, v in zip(num_digits, lev_dist_list)}

    # max level achieved - at any time
    perfect_runs = [k for k, v in dist_dict.items() if v == 0]
    if len(perfect_runs) > 0:
        max_level = np.max(perfect_runs)
    else:
        max_level = 0

    # calculate the best you did without any errors
    max_level_perfect = 0
    for ii in sorted(dist_dict.keys()):
        if dist_dict[ii] == 0:
            max_level_perfect = ii
        else:  # they failed
            break

    return {'transcript': transcript,
            'stim_times_start': stim_times_start,  # times the stimulus presentation ended
            'stim_times_end': stim_times_end,  # times the stimulus presentation ended
            'stim_digits': stim_digits,  # the list of digits as a string, one element per stimulus
            'responses': responses,  # subject's concatenated response, one per stimulus
            'responses_for_eval': responses_for_eval,  # if backwards, this is the response flipped horizontally
            'lev_dist_list': lev_dist_list,
            # the number of insertions, deletions or substitutions necessary to make the response match
            'levinstein_dist_dict': dist_dict,  # lev distance, where the key is the number of digits
            'set_dist_list': set_dist_list,  # true if all the correct numbers are listed, regardless of number order
            'dyslexic_swap_list': dyslexic_swap_list,  # true if a single horizontal swap error was made like: 125, 152
            'dyslexic_flip_list': dyslexic_flip_list,  # true if a single vertical flip error was made like, 16->19
            # todo: add in the parameter for how long the green bar lasted for
            'metrics': {
                'max_level_perfect': max_level_perfect,
                # the maximum number of digits achieved, where every other level below is correct
                'max_level': max_level,
                # the maximum number of digits ever completed correctly, even if an easier level was failed during this exam
                'next_exam_params': np.max([min_digits, max_level_perfect - 1])
                # one less than the level you got to in this round
            },

            'has_error': False}

