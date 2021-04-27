def gather_normal_prosaccades():
    """Using the file structure get all the normal data"""
    # todo: reformat to get these from the database
    # our concussion patient
    cidd = '09539aac-01f6-4332-ad94-9063b3160f48'

    data_folders = utils_df.get_data_folders('data/Good Data')
    results = utils_df.load_results(data_folders)
#     convert the results into a data frame
# convert the results dictionary into a dataframe
    cols = ['reaction_times', 'stim_starts', 'stim_ends', 'eye_starts', 'eye_ends', 'eye_fixs', 'max_speeds']
    all_results = []
    for k, v in results.items():
        df = pd.DataFrame.from_dict({k: v['prosaccade'][k] for k in cols})
        df['id'] = v['id']
        df['drinks'] = v['drinks']
        all_results.append(df)
    # get everything in one DF
    sdf = pd.concat(all_results)
    sdf['concussion'] = sdf['drinks'].apply(lambda w: 'concussed' if w == 'concussion' else 'normal')
    sdf['distance'] = np.round(10 * np.abs(sdf['stim_ends'] - sdf['stim_starts'])) / 10.0


    # quantify the errors
    sdf['error_vec'] = sdf['eye_fixs'] - sdf['eye_ends']
    sdf['error_mag'] = np.abs(sdf['error_vec'])

    # error in the direction of the saccade
    # dot product of stim direction vector - and error vec.
    sdf['stim_vec'] = sdf['stim_ends'] - sdf['stim_starts']
    stim_vec = sdf['stim_vec']
    sdf['overshoot'] = etl.complex_dot(sdf['error_vec'].values, sdf['stim_vec'].values)

    # calibrate people's positions
    sdf['stim_ends_r'] = np.around(sdf['stim_ends'], decimals=3)
    sdf['stim_starts_r'] = np.around(sdf['stim_starts'], decimals=3)
    stim_pos = np.unique(sdf['stim_ends_r'])
    calb_dict = dict()
    for pos in stim_pos:
        #     2D median
        z = sdf['eye_ends'][sdf['stim_ends_r'] == pos].values
        x, y = etl.geometric_median(np.array([z.real, z.imag]).T)
        calb_dict[pos] = x + 1j * y

    sdf['stim_ends_c'] = [calb_dict[pos] for pos in sdf['stim_ends_r']]
    sdf['stim_starts_c'] = [calb_dict[pos] for pos in sdf['stim_starts_r']]
    sdf['stim_vec_c'] = sdf['stim_ends_c'] - sdf['stim_starts_c']
    sdf['error_c'] = sdf['stim_ends_c'] - sdf['eye_ends']
    sdf['overshoot_c'] = etl.complex_dot(sdf['error_c'].values, sdf['stim_ends_c'].values)


    # for cnt, tmp in enumerate([.5, 1]):
    # I = sdf['distance'] == tmp
    norm_stims = stim_vec
    norm_eye_ends = sdf['error_c'] * np.exp(-1j * np.angle(stim_vec)) / np.abs(stim_vec)
    norm_reaction_times = sdf['reaction_times']
    norm_distances = sdf['distance']

    df = pd.DataFrame.from_dict({'norm_stims': norm_stims,
                                 'norm_eye_ends': norm_eye_ends,
                                 'norm_reaction_times': norm_reaction_times,
                                 'norm_distances': norm_distances})

    return calb_dict, df