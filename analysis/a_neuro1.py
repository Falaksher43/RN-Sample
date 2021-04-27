from analysis.a_general import *


CF_FILE_DICT = {"fruits": "config/listOfFruits.txt",
                "vegetables": "config/listOfVegetables.txt",
                "animals": "config/listOfAnimals.txt"}


# todo: do something fancy where the fruits classes inherit a common "food" class to reduce code duplication
class Fruits(object):
    name = 'fruits'

    def __init__(self, nlp, label='Fruit'):
        with open('config/listOfFruits.txt') as f:
            self.fruits = [line.replace('\n', '') for line in f]

        # initialise the matcher and add patterns for all fruits
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('FRUITS', None, *[nlp(c) for c in self.fruits])
        self.label = nlp.vocab.strings[label]

        # register extensions on the token
        Token.set_extension('is_fruit', default=False, force=True)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []

        for _, start, end in matches:
            # create Span for matched fruit and assign label
            entity = Span(doc, start, end)
            spans.append(entity)
            for token in entity:
                token._.set('is_fruit', True)

        doc.ents = list(doc.ents) + spans
        for span in spans:
            span.merge()

        return doc


class Vegetables(object):
    name = 'vegetables'

    def __init__(self, nlp, label='Vegetable'):
        with open('config/listOfVegetables.txt') as f:
            self.vegetables = [line.replace('\n', '') for line in f]

        # initialise the matcher and add patterns for all fruits
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('VEGETABLES', None, *[nlp(c) for c in self.vegetables])
        self.label = nlp.vocab.strings[label]

        # register extensions on the token
        Token.set_extension('is_vegetable', default=False, force=True)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []

        for _, start, end in matches:
            # create Span for matched vegetable and assign label
            entity = Span(doc, start, end)
            spans.append(entity)
            for token in entity:
                token._.set('is_vegetable', True)

        doc.ents = list(doc.ents) + spans
        for span in spans:
            span.merge()

        return doc


def group_words(words_list, category_file):
    category_df = pd.read_csv(category_file, header=None, names=['name'])
    # count how many words in each item
    category_df['nwords'] = category_df['name'].apply(lambda w: len(w.split(" ")))

    # group the word list by number of words, and make them into a set
    category_series = category_df.groupby('nwords')['name'].apply(set)

    # order it from most words to least (important)
    category_series = category_series.sort_index(ascending=False)

    words_list = words_list.values
    words_list = list(words_list)
    for n, cur_set in category_series.items():
        words_list2 = []
        ii = 0
        if ~(n == 1):
            while ii <= len(words_list) - n:
                next_n_words = " ".join(words_list[ii:ii + n])
                if next_n_words in cur_set:
                    if n > 1:
                        print('combining: ' + next_n_words)
                    words_list2.append(next_n_words)
                    ii += n
                else:
                    words_list2.append(words_list[ii])
                    ii += 1
            print("ASSIGNMENT ISSUE WITH N?", n)
            print(words_list2)

    # add on the last bits
    for ii in np.arange(ii, len(words_list)):
        words_list2.append(words_list[ii])

    words_list = words_list2

    return words_list



def process_categoryfluency(time_df, audio_dict):

    category = ''
    for key in audio_dict:
        if 'filepath' in key:
            category = key.split('_')[0]

    next_category = get_next_exam_param(category, 'categoryfluency')

    response = audio_dict['transcript']
    response = response.loc[response['transcript'].apply(type) == str]
    response['transcript'] = response['transcript'].apply(lambda x: x.lower())
    response['start_time'] = response['start_time'].astype('float')
    response['end_time'] = response['end_time'].astype('float')
    response['duration'] = response['end_time'] - response['start_time']

    # IDENTIFY VALID RESPONSES
    file = CF_FILE_DICT[category]
    with open(file) as f:
        list_to_check = f.read().splitlines()
        list_to_check = [x.lower() for x in list_to_check]

    with open('config/nouns.txt') as nounfile:
        list_of_nouns = nounfile.read().splitlines()

    # check appropriate list/dict to see if the word is in the category
    found_category = response.transcript.apply(lambda x: x in list_to_check)
    response['found_category'] = found_category

    # check for pairwise words ('summer squash')
    for row1, row2 in more_itertools.pairwise(response.iterrows()):
        if row1[1]['found_category'] == False & row2[1]['found_category'] == False:
            possibly_correct = row1[1]['transcript'] + ' ' + row2[1]['transcript']
            if possibly_correct in list_to_check:
                response.loc[
                    response['start_time'] == row1[1]['start_time'], 'found_category'] = True

                # combine two words to make 1 row --> makes for better plotting and counting
                response.loc[response['start_time'] == row1[1][
                    'start_time'], 'transcript'] = possibly_correct

                # removing second word of pairwise combo from dataframe
                response = response.loc[~(response['start_time'] == row2[1]['start_time'])]

    # check for duplicates, accounting for duplicates that are plural
    # p.singular_noun() returns False if the word is already singular
    p = inflect.engine()
    singular_words = response.transcript.apply(lambda x: p.singular_noun(x) if p.singular_noun(x) else x)
    category_duplicates = singular_words.duplicated()
    response['category_duplicates'] = category_duplicates

    # check to see if word is a noun. this is used for determining intrusions
    nouns = singular_words.apply(lambda n: n in list_of_nouns)
    response['noun'] = nouns
    response['is_intrusion'] = (response.noun & ~response.found_category)

    # this would be used to check the singular words in case we just don't have the plurals in the list
    # but this fails with words like asparagus due to inflect.engine()
    # check_words = singular_words.apply(lambda w: w in list_to_check)
    # response['found_category'] = check_words

    # RESULTS
    results = dict()
    results['start'] = response.iloc[0]['start_time']
    results['finish'] = response.iloc[-1]['end_time']
    results['duration'] = results['finish'] - results['start']
    results['responses'] = response
    results['category'] = category
    results['is_category'] = response['found_category']
    results['category_duplicates'] = response['category_duplicates']
    results['is_intrusion'] = response['is_intrusion']
    results['next_category'] = next_category

    results['status'] = True

    metrics = dict()
    metrics['num_correct'] = (results['is_category'] & ~results['category_duplicates']).sum()
    metrics['num_repeats'] = (results['is_category'] & results['category_duplicates']).sum()
    metrics['num_intrusions'] = results['is_intrusion'].sum()
    metrics['next_exam_params'] = results['next_category']

    return {'data': results,
            'metrics': round_metrics(metrics),
            'has_error': False}


def process_letterfluency(time_df, audio_dict):
    for key in audio_dict:
        if 'filepath' in key:
            audio_key = key.split('_')[0]

    letter = audio_key[-1].lower()

    next_letter = get_next_exam_param(letter, 'letterfluency')

    response = audio_dict['transcript']
    response = response.loc[response['transcript'].apply(type) == str]
    response['transcript'] = response['transcript'].apply(lambda x: x.lower())
    response['start_time'] = response['start_time'].astype('float')
    response['end_time'] = response['end_time'].astype('float')
    response['duration'] = response['end_time'] - response['start_time']

    correct_letter = response.transcript.apply(lambda x: x[0] == letter)
    duplicate_word = response.transcript.duplicated()

    with open('config/nouns.txt') as nounfile:
        list_of_nouns = nounfile.read().splitlines()

    p = inflect.engine()
    singular_words = response.transcript.apply(lambda x: p.singular_noun(x) if p.singular_noun(x) else x)
    nouns = singular_words.apply(lambda n: n in list_of_nouns)
    is_intrusion = (nouns & ~correct_letter)

    # RESULTS
    results = dict()
    results['start'] = response.iloc[0]['start_time']
    results['finish'] = response.iloc[-1]['end_time']
    results['duration'] = results['finish'] - results['start']
    results['responses'] = response
    results['letter'] = letter
    results['correct_letter'] = correct_letter
    results['duplicate_word'] = duplicate_word
    results['is_intrusion'] = is_intrusion
    results['next_letter'] = next_letter

    metrics = dict()
    metrics['num_correct'] = (results['correct_letter'] & ~results['duplicate_word']).sum()
    metrics['num_repeats'] = (results['correct_letter'] & results['duplicate_word']).sum()
    metrics['num_intrusions'] = results['is_intrusion'].sum()
    metrics['next_exam_params'] = results['next_letter']

    return {'data': results,
            'metrics': metrics,
            'has_error': False}



# functions for process_trailmaking:
def index_to_labeled_stops(df):
    """
    Take the selected index and label errors, and also return the values at the stops (for location)
    :param selected_index:
    :return:
    """
    time = df['Total Time'].values
    selected_index = df['Selected Index'].values + 1

    stop_values = []  # the index of the circle you are at
    stop_labels = []  # 0: correct, 1: repeat of last one, 2: error
    stop_times = []  # the time on the clock when you first hit that stop
    stop_durations = []  # the duration the person spent at the dot
    targeted_idxs = []  # the index at the moment you first hit that stop
    targeted_idxs2 = []  # the index at the moment you first hit that *last stop

    correct_index = 1
    last_index = -2  # so when it starts it won't be
    starts_ends = etl.start_and_ends(selected_index)
    for cur in starts_ends:
        start, endd = cur
        stop_values.append(selected_index[start + 1])
        stop_durations.append(time[endd] - time[start])
        stop_times.append(time[start + 1])
        targeted_idxs.append(endd + 1)
        targeted_idxs2.append(start + 1)
        if selected_index[start + 1] == correct_index:
            stop_labels.append(0)
            correct_index += 1
            last_index = correct_index - 1
        elif selected_index[start + 1] == last_index:
            stop_labels.append(1)
        else:
            stop_labels.append(2)

    return np.array(stop_values), \
           np.array(stop_labels), \
           np.array(stop_times), \
           np.array(stop_durations), \
           np.array(targeted_idxs), \
           np.array(targeted_idxs2)


def prep_x(x, y, z):  # (tricky to get this inside of wrap_pca)
    return np.array([x, y, z]).T


class wrap_pca:

    def fit(x, y, z):
        pca = PCA(n_components=3)
        cur_decomp = pca.fit(prep_x(x, y, z))
        return cur_decomp

    def transform(x, y, z, cur_decomp):
        return cur_decomp.transform(prep_x(x, y, z)).T


def get_stim_decomp(df):
    # train the PCA model on only real values
    I = (df['Selected Index'] + 1).astype(bool)
    x = df[I]['Stimulus 1 Position X']
    y = df[I]['Stimulus 1 Position Y']
    z = df[I]['Stimulus 1 Position Z']
    I2 = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    stim1 = np.array([x[I2], y[I2], z[I2]])
    cur_decomp = wrap_pca.fit(*stim1)

    # measure how planar the stimuli are
    stim_ = wrap_pca.transform(*stim1, cur_decomp)
    stim_flatness = np.std(stim_[2, :])

    # set the nan values to zero so PCA can be applied
    x = df['Stimulus 1 Position X'].fillna(0)
    y = df['Stimulus 1 Position Y'].fillna(0)
    z = df['Stimulus 1 Position Z'].fillna(0)
    stim = np.array([x, y, z])
    stim2 = wrap_pca.transform(*stim, cur_decomp)
    return stim2, cur_decomp, stim_flatness


def get_laser_projection(df):
    stim2, cur_decomp, stim_flatness = get_stim_decomp(df)
    origin = wrap_pca.transform([0], [0], [0], cur_decomp)

    if 'Laser Hit Point X' in df.columns:
        x = df['Laser Hit Point X']
        y = df['Laser Hit Point Y']
        z = df['Laser Hit Point Z']
        laser = np.array([x, y, z])
        laser2 = wrap_pca.transform(*laser, cur_decomp)
        v_new = laser2

    else:
        # this is the old system computing quaternions
        x = df['Laser Position X']
        y = df['Laser Position Y']
        z = df['Laser Position Z']
        laser = np.array([x, y, z])
        laser2 = wrap_pca.transform(*laser, cur_decomp)

        if 'Pointer Rotation W' in df.columns:
            Q = df[['Pointer Rotation W', 'Pointer Rotation X', 'Pointer Rotation Y', 'Pointer Rotation Z']].values
        else:
            Q = df[['Laser Rotation W', 'Laser Rotation X', 'Laser Rotation Y', 'Laser Rotation Z']].values
            logger.warning(
                "Distance accuracy metrics innacurate for missing Laser Hit Point & Pointer Rotation missing.")
        Qs = [Quaternion(q).normalised for q in Q]

        # unit vectors in the direction of the quaternion
        V = np.array([w.rotate([0, 0, 1]) for w in Qs])
        # Do PCA to get it in units of the flat stimulus
        V2 = wrap_pca.transform(*V.T, cur_decomp)

        #     the quaternion vector - in the new coordinate system, centered on the origin
        v_diff = V2.T - origin.T

        # project the laser from the origin to the flat surface
        C = np.array([v[2] / w[2] for v, w in zip(laser2.T, v_diff)])
        v_new = laser2 - (v_diff * C.reshape(len(laser2[0]), 1)).T

    return v_new, stim2, origin, stim_flatness


def process_trailmaking2(time_df, audio_dict):
    return process_trailmaking(time_df, audio_dict)


def process_trailmaking(time_df, audio_dict):
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
        new_index = np.array(new_index) + 1
        return new_index

    t, z = utils_df.get_t_z(time_df)
    _, pupilz = utils_df.get_t_pupilz(time_df)

    # separate the longer instructions period
    start_idx, _ = utils_df.get_exam_start(time_df)
    time_df = time_df.iloc[start_idx:]

    time = time_df['Total Time'].values

    # needed for plot_trail_accuracy
    stop_values, stop_labels, stop_times, stop_durations, targeted_idx, targeted_idx2 = index_to_labeled_stops(time_df)
    selected_index = time_df['Selected Index'].values + 1

    # count the errors:

    I = stop_labels == 0
    highest_level = sum(I)

    if highest_level == 0:
        # if you didn't even hit a single bubble
        good_data = False
        highest_level = np.nan
        active_time = np.nan
        total_time = np.nan
        error_count = np.nan
        repeat_count = np.nan
        stop_values = np.nan
        stop_labels = np.nan
        stop_times = np.nan
        targeted_idx = np.nan
        search_durations = np.nan

    else:
        good_data = True

        search_durations = np.diff([time_df['Total Time'].values[0], *stop_times[I]])

        active_time = max(stop_times) - min(stop_times)
        total_time = max(stop_times) - min(time)

        I = stop_labels == 2
        error_count = sum(I)

        I = stop_labels == 1
        repeat_count = sum(I)

    if sum(I) > 0:
        # quantify the errors:
        v_new, stim2, origin, stim_flatness = get_laser_projection(time_df)

        pointer = v_new[0, :] + 1j * v_new[1, :]
        stimulus = stim2[0, :] + 1j * stim2[1, :]

    else:
        pointer = []
        stimulus = []
        stim_flatness = 100

    # checp for too many stimulus values which could screw up the projection data
    if len(np.unique(np.around(stimulus, decimals=3))) > len(np.unique(stop_values)) + 1:
        stim_flatness = 100

    metrics = {
        'total_time': total_time,
        'error_count': error_count,
        'repeat_count': repeat_count,
        'num_correct': highest_level,
        'next_exam_params': None
    }

    results = {
        'good data': good_data,
        'number correct': highest_level,
        'active time': active_time,  # time spent actively searching
        'total time': total_time,  # time from start of exam until you tap the last target
        'average speed': highest_level / active_time,
        'error count': error_count,
        'error rate': error_count / highest_level,
        'repeat count': repeat_count,
        'repeat rate': repeat_count / highest_level,
        'stop values': stop_values,
        'stop labels': stop_labels,
        'stop times': stop_times,
        'targeted idxs': targeted_idx,
        'targeted idxs2': targeted_idx2,
        'search durations': search_durations,  # duration between successive correct answers, including the first

        'stim flatness': stim_flatness,
        'valid positions': stim_flatness < 0.5,
        'stimulus': stimulus,
        'pointer': pointer,
        'z': z,
        'pupilz': pupilz,
        'selected index': selected_index,
        't': t,
        'time': time,
        'start_idx': start_idx,
        'stop labels': stop_labels,
        'new index': get_new_index(selected_index),

        'metrics': round_metrics(metrics),

        'has_error': False
    }

    return results

