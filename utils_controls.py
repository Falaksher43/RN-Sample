import pandas as pd
import numpy as np
import time
import datetime
import pickle
import utils_db
import utils_metrics
import json

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

CONTROL_SUBJECT_QTY = 100

# contains the mapping of exams to their controls files
with open('./.control_data/controls_file_mapping.json') as f:
    EXAM_CONTROL_FILE_DICT = json.load(f)


def filter_most_recent_exams(visit_exams_df):
    """
    Get the most recent exam of each type per user
    :return:
    """
    host = utils_db.get_host()

    # remove anyone who is a test subject if on production
    if host == 'production':
        test_subjects = utils_metrics.get_test_subjects()
        visit_exams_df = visit_exams_df[visit_exams_df['subject_id'].apply(lambda w: w not in test_subjects)]

    # keep only the exams that don't have any processing errors
    visit_exams_df = visit_exams_df.loc[visit_exams_df['has_error'] == False]

    # pick just the most recent exam for each visit exam type for each subject
    # sort the exams by date
    visit_exams_df = visit_exams_df.sort_values('created_date_visit', axis=0)

    # just pick the last exam that each user took
    visit_exams_df = visit_exams_df.groupby(['subject_id', 'exam']).last().reset_index()

    return visit_exams_df

def generate_control_population_pickle(exam):

    host = utils_db.get_host()
    if host == 'staging':
        raise Exception("Cannot generate controls file from staging")

    # get all visit_exams
    visit_exams_df = utils_db.get_visit_exams(get_visits(n=np.inf))

    # get the most recent exam of each type per user
    subset = filter_most_recent_exams(visit_exams_df)

    # filter out the exam that we're looking for
    subset = subset[subset['exam'] == exam]

    # load the processed data for all of these visits
    all_processed_data = utils_db.load_visit_exams_processed(subset)

    # get the date so that we can know when this file was generated
    today = datetime.date.today()
    d = today.strftime("%b-%d-%Y")

    # save as pickle file in the control_data folder
    filename = './.control_data/' + exam + '_ProcessedData_' + d + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(all_processed_data, f)


def update_controls_processed_pickle(exam):

    host = utils_db.get_host()
    if host == 'staging':
        raise Exception("Cannot update controls from staging")

    if exam not in EXAM_CONTROL_FILE_DICT:
        logger.warning('No control pickle file to update for exam. Please check exam and generate pickle if necessary')
    else:
        # open the old control file
        filepath = './.control_data/' + EXAM_CONTROL_FILE_DICT[exam]
        exams_on_file = pickle.load(open(filepath, "rb"))

        # take out the processed column so that we can compare to get_visit_exams
        without_processed_column = exams_on_file.iloc[:, 0:-1]

        # get all visit_exams
        current_visit_exams_df = utils_db.get_visit_exams(get_visits(n=np.inf))

        # get the most recent exam of each type per user
        subset = filter_most_recent_exams(current_visit_exams_df)

        # filter out the exam that we're looking for
        subset = subset[subset['exam'] == exam]

        # concatenate them together and drop any rows that are exactly the same
        check_differences = pd.concat([without_processed_column, subset], sort=False).drop_duplicates(keep=False)

        # drop older versions of the exam for each subject
        check_differences = check_differences.sort_values('visit_exam_id', ascending=False).drop_duplicates(
            subset=['subject_id'], keep='last').sort_index()

        # for each row of the new results check to see if the subject_id exists in the old one
        # if so, check to see if it's an older exam and append the new one to list of exams to run load_processed
        new_exams = pd.DataFrame()
        for i, result in check_differences.iterrows():
            if result['subject_id'] in exams_on_file['subject_id'].values:
                subject_in_old = exams_on_file.loc[exams_on_file['subject_id'] == result['subject_id']]
                #                 print(result['created_date'], subject_in_old.iloc[0]['created_date'])
                if result['created_date'] != subject_in_old.iloc[0]['created_date']:
                    new_exams = new_exams.append(result)
            else:
                new_exams = new_exams.append(result)

        # load the new processed_data
        new_processed = utils_db.load_visit_exams_processed(new_exams)

        # concatenate the new and old exams and drop older versions of the exam for each subject
        old_and_new_exams = pd.concat([exams_on_file, new_processed])
        old_and_new_exams = old_and_new_exams.sort_values('visit_exam_id', ascending=False).drop_duplicates(
            subset=['subject_id'], keep='last').sort_index()

        # TODO: FIGURE OUT WHAT TO DO WITH THE OLD CONTROL FILES (need to keep for FDA probably)

        return old_and_new_exams

def load_processed_controls_from_pickle(subj_series, n=CONTROL_SUBJECT_QTY, year_range=20, max_controls=False):
    exam = subj_series['exam']
    if exam in EXAM_CONTROL_FILE_DICT.keys():
        filepath = './.control_data/' + EXAM_CONTROL_FILE_DICT[exam]
        visit_exams_df = pickle.load(open(filepath, "rb"))
        visit_exams_df = visit_exams_df.loc[visit_exams_df['subject_id'] != subj_series['subject_id']]

        if max_controls:
            control_df = visit_exams_df

        else:
            if (subj_series['date_of_birth'] is None) or (subj_series['gender'] is None):
                # do not return any controls until we know the person's date of birth
                control_df = pd.DataFrame()
                logger.warning("missing DOB or Gender for: " + str(subj_series['subject_id']))

            else:
                visit_exams_df.dropna(subset=['date_of_birth'], inplace=True)
                visit_exams_df.dropna(subset=['gender'], inplace=True)

                # filter
                filter_gender = visit_exams_df.gender == subj_series.gender

                # calculate age of subj_series and filter based on year_range
                subj_series_age = subj_series['created_date_visit'] - subj_series['date_of_birth']
                visit_exams_df['age'] = visit_exams_df['created_date_visit'] - visit_exams_df['date_of_birth']
                filter_age = (visit_exams_df.age > subj_series_age - datetime.timedelta(days=year_range * 365.25)) & \
                             (visit_exams_df.age < subj_series_age + datetime.timedelta(days=year_range * 365.25))
                visit_exams_df = visit_exams_df[filter_gender & filter_age]

                # drop the age column once we're done using so nothing farther downstream breaks
                visit_exams_df.drop(['age'], axis=1, inplace=True)

                control_df = visit_exams_df.iloc[:n]
    else:
        control_df = pd.DataFrame()
        logger.warning('No control data pickle file. Please check exam name or generate file.')

    controls_and_subject_processed_df = control_df.append(subj_series)

    return controls_and_subject_processed_df