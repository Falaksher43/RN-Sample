
import os
import numpy as np
import pandas as pd
from utils_df import clean_df, count_blinks, flip_x
from matviz.matviz import etl
from io import StringIO, BytesIO
import json
import boto3
import time
import urllib
import psycopg2
import pickle
import datetime
import pathlib
# from pydub import AudioSegment
import pydub
import parselmouth
import librosa
from parselmouth.praat import call
import uuid
import time

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

from botocore.exceptions import ClientError

# it is used, but with an eval so that it appears unused by the linter
import analysis

BUCKET_URL = 'https://reactneuro-data.s3.amazonaws.com/'
BUCKET_DATA = 'reactneuro-data'
PATH_AWS_PRODUCTION = 'production-alpha'
PATH_AWS_STAGING = 'internal/testing'
CONFIG_FILE = '.aws/production_staging_config.json'

TEST_LOCATION_NAMES = ['Test Site one', 'Test Site Two']

def map_question_to_name():
    return {
                1: "eye color",
                2: "concussion",
                3: "impairment",
                4: "exam comments",
                5: "glasses",
                6: "administrator",
                7: "general comments",
                8: "q_gender",
                9: "q_email"
            }


def pd_get_id_map(cur_table):
    """
    Get a mapping from ID to name for a given table
    :param cur_table: the table name
    :return: dictionary of name to id
    """

    query = """
        SELECT name, id FROM dbo.{};
        """.format(cur_table)

    df = pd_read_sql_conn(query)

    return etl.dictify_cols(df[['name', 'id']])


def pd_read_sql_conn(query):
    """
    open and close the database connection and read SQL with pandsas in the middle
    :param query: string to read in sql
    :return: dataframe
    """
    # connect to the prod database
    conn = get_conn()
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def conditions_to_sql(conditions=[], **kwargs):
    """
    this is used only internally to help write sql queries
    todo: make this a class and exclude this function
    :param conditions: array with SQL condition code, optional
    :param kwargs: columns
    :return:
    """
    # interpret general user entered parameters
    for k, v in kwargs.items():
        conditions += [k + "='{}'\n".format(str(v))]

    if len(conditions) > 0:
        query = "WHERE " + "AND ".join(conditions)
    else:
        query = ""

    return query


def unstack_question_responses(df, key_col, exclude_col):
    """Unstack so that each question has a column, and each answer is in the row"""

    df = df[df[exclude_col].apply(lambda w: w is None)]

    df_mult_idx = df[[key_col, 'question_id', 'text']].set_index([key_col, 'question_id'], inplace=False)
    df_refact = df_mult_idx.unstack()['text'].reset_index()
    df_refact.columns.name = None

    return df_refact

def get_visits(n=100, created_since=False, visit_id=None, **kwargs):
    """
    Load everything about visits, default behaviour is to get the 100 most recent visit
    :param n: the number of recent visits to return
    :param created_since: date, only visits past that date will be returned
    : kwargs, can be any column in the visits table
         visit_id	user_id	location_id	device_id	s3_folder	created_date
    :return: data frame where each row is a visit


    example:
        get_visits(n=20, user_id='fish', created_date="jan 1 2015")

    """

    # query = "SELECT * FROM dbo.visits\n"
    # get the user information
    # query = """
    #             SELECT visits.*, responses.text AS concussion, users.date_of_birth, users.gender, users.first_name, users.last_name
    #               FROM dbo.visits as visits
    #
    #               LEFT JOIN dbo.question_responses as responses
    #                 ON visits.id = responses.visit_id
    #
    #               LEFT JOIN dbo.users as users
    #                 ON visits.user_id = users.id
    #
    #             WHERE responses.question_id=2
    #         """
    query = """
                SELECT visits.id, 
                        visits.device_id, 
                        visits.subject_id,
                        visits.location_id, 
                        visits.s3_folder, 
                        visits.comments, 
                        visits.created_date, 
                        visits.reports_processed, 
                        visits.reports_processing, 
                        visits.reports_json, 
                        subjects.date_of_birth, 
                        subjects.gender 
                  FROM dbo.visits as visits
                
                  LEFT JOIN dbo.subjects as subjects
                    ON visits.subject_id = subjects.id
            """
    conditions = []

    if created_since: # only order the data if you want the most recent ones
        conditions += ["created_date>'{}'\n".format(str(created_since))]

    if visit_id is not None:
        conditions += ["visits.id in {}\n".format(format_list_or_int_to_sql(visit_id))]

    query += conditions_to_sql(conditions=conditions, **kwargs)

    # query += conditions_to_sql(conditions=conditions, **kwargs).replace("WHERE", "AND")

    if n != np.inf: # only order the data if you want the most recent ones
        query += "ORDER BY created_date desc \nLIMIT {}".format(str(n))

    logger.debug(query)

    df = pd_read_sql_conn(query)

    visit_df = df.rename(columns={"id": "visit_id",
                                  "comments": "comments_visit"})

    # todo: integrate questions from different structure in db (stored as digital exam metrics, etc.)
    # # pull in the user table information
    # responses_subject = get_question_responses_by_subject(visit_df.subject_id.unique())
    # if len(responses_subject) > 0:
    #     responses_subject = responses_subject.drop_duplicates(subset=['question_id', 'subject_id', 'visit_id'], keep='last')
    #     responses_subject = unstack_question_responses(responses_subject, 'subject_id', 'visit_id')
    #     visit_df = pd.merge(visit_df, responses_subject, how='left', on='subject_id')
    #
    # # pull in the visit table information
    # responses_visit = get_question_responses_by_visit(visit_df.visit_id.values)
    # if len(responses_visit) > 0:
    #     responses_visit = responses_visit.drop_duplicates(subset=['question_id','visit_id'], keep='last')
    #     responses_visit = unstack_question_responses(responses_visit, 'visit_id', 'subject_id')
    #     visit_df = pd.merge(visit_df, responses_visit,  how='left', on='visit_id')
    #
    # # visit_df = visit_df.rename(columns=map_question_to_name())
    #
    # # remove q_email and q_gender if they exist
    # to_drop = ['q_gender', 'q_email']
    # to_drop = [w for w in to_drop if w in visit_df]
    # if len(to_drop) > 0:
    #     visit_df = visit_df.drop(columns=to_drop)
    #
    # # rename columns corresponding to question numbers (16, 17, etc.) to strings ("q_16")
    # rename_dict = {name: "q_" + str(name) for name in visit_df.columns if type(name) is int}
    # visit_df = visit_df.rename(columns=rename_dict)

    return visit_df


# todo: Some code duplication going on here
def get_question_responses_by_visit(visit_ids):
    """Get all the question responses for the visit IDs we need"""
    query = """
                SELECT *
                  FROM dbo.question_responses
                  WHERE visit_id IN {}
            """.format(format_list_to_sql(visit_ids))

    df = pd_read_sql_conn(query)

    return df


def format_list_or_int_to_sql(x):
    """
    covert an ID or a list of IDs and convert it to SQL for a 'in (tuple)' format
    :param x: list or singular visit_id
    :return: string that can be used for sql
    """

    if type(x) not in [list, np.ndarray]:
        x = [x]

    return format_list_to_sql(x)



def format_list_to_sql(x):

    if len(x) == 0:
        query = "('none')"
    elif len(x) == 1:
        query = "('" + str(x[0]) + "')"
    else:
        query = str(tuple(x))

    return query




def get_question_responses_by_subject(subject_ids):
    """Get all the question responses for the user IDs we need"""
    query = """
                SELECT *
                  FROM dbo.question_responses
                  WHERE subject_id IN {}
            """.format(format_list_to_sql(subject_ids))

    # question_id, subject_id, visit_id, text

    df = pd_read_sql_conn(query)

    return df


def get_visit_exams(visit_df, **kwargs):
    """
    load the exams and exam metadata for one or more exams
    :param visit_df: the output from 'get_visits'
    :return: data where each row is an exam, and the data from vistis_df is maintained
    """

    # rename created date here so it doesn't conflict with the visit_exam created date
    visit_df = visit_df.rename(columns = {"created_date": "created_date_visit"})

    query = "SELECT * FROM dbo.visit_exams\n"


    # handle user inputs
    if type(visit_df) == pd.core.frame.DataFrame:
        # raise Exception("Cannot handle more than one visit at at time (yet)")
        visit_ids = visit_df['visit_id'].values

        # only get the visit exams that are in our visit ID
        query += "WHERE visit_id in {} ".format(format_list_to_sql(visit_ids))

        # add in any conditions for those visit exams (which exam do you want?)
        query += conditions_to_sql(**kwargs).replace("WHERE", "AND")

        logger.debug(query)
        df = pd_read_sql_conn(query)

        # Remove duplicate rows
        df = df.drop_duplicates(subset=['visit_id', 'exam_id'], keep='last')

        # change column names to prepare for the merge
        df = df.rename(columns={"id": "visit_exam_id", "comments": "comments_exam"})

        if len(df) == 0:
            logger.warning('No exams for visit_id = {}'.format(str(visit_ids)))
            return df
        else:
            visit_exams_df = pd.merge(df, visit_df, how='left', on=['visit_id', 'subject_id', 'location_id'])


    elif type(visit_df) == pd.core.series.Series:
        visit_id = visit_df['visit_id']
        # find the exam IDs for this
        query += "WHERE visit_id = '{}'".format(visit_id)

        logger.debug(query)

        df = pd_read_sql_conn(query)

        df = df.rename(columns={"id": "visit_exam_id", "comments": "comments_exam"})

        # merge in the data from the visit_df
        visit_exams_df = pd.merge(df, visit_df.to_frame().transpose(), how='left', on='visit_id')

    else:
        raise Exception("get_visit_exam accepts a 'visit_df' returned from get_visits")

    visit_exams_df = visit_exams_df.reset_index()

    visit_exams_df['age'] = (visit_exams_df['created_date_visit'] - visit_exams_df['date_of_birth']).apply(
        lambda w: w.days / 365.25)

    return human_readable_df(visit_exams_df)


def get_visit_exams_by_id(visit_exam_id):
    """
    This function goes from visit_exam_id, to visits (for the device information etc.)
    Then back to visit exams
    Then filters on those visit exams to get just the ids we want again
    """
    # get the visits from the visit exams table
    query = """
            SELECT * 
                FROM dbo.visit_exams
                WHERE exam_version IS NOT NULL
    """.format(visit_exam_id)

    if  hasattr(visit_exam_id, '__len__'):
        query += "AND id IN {}""".format(format_list_to_sql(visit_exam_id))
    else:
        query += "AND id = '{}'".format(visit_exam_id)
        visit_exam_id = [visit_exam_id]

    df = pd_read_sql_conn(query)

    # get the visits
    df2 = get_visits(visit_id=df['visit_id'].values)

    # get the visit_exams
    df3 = get_visit_exams(df2)
    df4 = df3[df3['visit_exam_id'].apply(lambda w: w in visit_exam_id)]
    df4 = human_readable_df(df4)
    return df4

def load_visit_exams_audio(visit_exams_df, params={'overwrite_audio': False}):
    """
    Load the audio data and add as a column
    :param visit_exams_df:
    :return: same format as visit_exams_df, but with added audio column
    """

    def visit_exam_to_audio(subj_series):
        """

        :param visit_exam_id:
        :return: dictionary with audio
        """
        s3_path = subj_series['s3_folder']
        exam = subj_series['exam']
        visit_exam_id = int(subj_series['visit_exam_id'])
        error_message = None

        if type(s3_path) is not str:
            logger.warning("loading audio failed for visit_exam_id: {} due to s3_path format".format(visit_exam_id))
            error_message = 'loading audio failed due to s3_path format'
            audio_dict = None
        else:
            fullpath = s3_path + '/' + exam + '/'
            prefix = fullpath.replace(BUCKET_DATA + "/", "")

            audio_dict = dict()

            audio_dict['s3_path'] = s3_path

            s3_client = boto3.client('s3', **read_credentials())
            response = s3_client.list_objects_v2(Bucket=BUCKET_DATA, Prefix=prefix)
            #
            if 'Contents' in response:
                for key in response['Contents']:
                    if '.wav' in key['Key']:
                        # gets the key to the s3 object
                        filekey_audio = key['Key']

                        # READ OBJECT
                        obj = s3_client.get_object(Bucket=BUCKET_DATA, Key=filekey_audio)

                        # UNWRAP OBJECT
                        data = BytesIO(obj['Body'].read())

                        # PYDUB AUDIOSEGMENT SEEMS TO RESCALE AUDIO ...
                        # import ffmpeg
                        try:
                            audio = pydub.AudioSegment.from_file(data)
                        except pydub.exceptions.CouldntDecodeError:
                            audio = None
                            logger.warning("Audio file badly formatted: " + filekey_audio)
                            error_message = "Audio file badly formatted"

                        # gets the name of the file without the path before it, e.g. 'fruits.wav'
                        exam_file = os.path.basename(filekey_audio)

                        # removes the extension so we can put it in the dict
                        exam = os.path.splitext(exam_file)[0]

                        # audio_dict[result] = BUCKET_DATA + '/' + filekey
                        audio_dict[exam] = audio
                        audio_dict[exam + '_filepath'] = filekey_audio

                # getting the transcript if it exists on s3 already and adding to audio_dict
                # this helps us avoid re-transcribing audio each time we re-process
                transcript_exists = False
                for key in response['Contents']:
                    if 'transcript' in key['Key']:
                        transcript_exists = True
                        filekey_transcript = key['Key']


                # if the transcript exists and we don't want it overwritten, load it. Otherwise, transcribe using
                # Amazon Transcribe
                if transcript_exists and not params['overwrite_audio']:
                    transcript_obj = s3_client.get_object(Bucket=BUCKET_DATA, Key=filekey_transcript)
                    transcript_data = transcript_obj['Body'].read()
                    transcript = pickle.loads(transcript_data)

                else:
                    # job name requires a UUID since job names can never be duplicated (Transcribe is annoying this way)
                    job_name = str(visit_exam_id) + '_' + exam + '_' + str(uuid.uuid4())
                    job_uri = BUCKET_URL + prefix + exam + '.wav'
                    transcript = get_transcription_amazon(job_name=job_name, job_uri=job_uri)

                    # make sure that the transcription completed successfully
                    if transcript is None:
                        logger.warning('Transcription empty, empty pickle file created for visit_exam_id: {}'.format(visit_exam_id))
                        transcript = pd.DataFrame(columns=['start_time', 'end_time', 'transcript', 'alternatives'])

                    # write transcript to a pickle file
                    with open('transcript.pickle', 'wb') as f:
                        pickle.dump(transcript, f)
                    transcript_filepath = prefix + 'transcript.pickle'

                    # upload file to appropriate folder in s3 and remove local copy
                    s3_client.upload_file(Filename='transcript.pickle', Bucket=BUCKET_DATA, Key=transcript_filepath)
                    os.remove('./transcript.pickle')

                audio_dict['transcript'] = transcript

        return audio_dict, error_message

    # print(visit_exams_df)
    all_data = visit_exams_df.apply(visit_exam_to_audio, axis=1)
    visit_exams_df[['audio', 'error_audio']] = pd.DataFrame(all_data.to_list(), index=visit_exams_df.index)

    # # filter out any empty dataframes
    # visit_exams_df = visit_exams_df[visit_exams_df['audio'].apply(lambda w: len(w)>0)]
    # visit_exams_df = visit_exams_df[visit_exams_df['audio'].apply(lambda w: w is not None)]

    # reset the index after these elements were deleted
    visit_exams_df = visit_exams_df.reset_index(drop=True)

    return visit_exams_df


def load_visit_exams_timeseries(visit_exams_df):
    """
    Load the eye movement timeseries data and add it as a column
    :param visit_exams_df:
    :return: same format as visit_exams_df, but with an added timeseries column
    """
    conn_ts = get_conn()

    def visit_exam_to_timeseries(visit_exam_id):
        """
        Convert a visit exam into a standardized dataframe

        Note: it uses id2column as a global variable to save time

        :param visit_exam_id: integer
        :return: timeseries dataframe
        """

        visit_exam_id_int = int(visit_exam_id)
        if visit_exam_id_int != visit_exam_id:
            raise Exception("Non integer visit_exam_id found for visit_exam_id: " + str(visit_exam_id))
        query = """
        SELECT * FROM dbo.visit_exam_columns
        WHERE visit_exam_id = '{}';
        """.format(visit_exam_id_int)


        # query = """
        # SELECT DISTINCT ON (column_id)
        # id, visit_exam_id, column_id
        # FROM   dbo.visit_exam_columns
        # WHERE visit_exam_id = '{}'
        # """.format(visit_exam_id)

        all_cols_df = pd.read_sql(query, conn_ts)
        error_message = None
        if len(all_cols_df) == 0:
            logger.info("Empty data for visit_exam_id: " + str(visit_exam_id))
            error_message = 'Missing all data'
            df = pd.DataFrame()

        else:
            to_df = []
            for index, row in all_cols_df[['column_id', 'data']].iterrows():
                cur_col = id2column[row['column_id']]
                cur_bytes = row['data']
                cur_str = bytes(cur_bytes).decode('utf-8')
                cur_df = pd.read_csv(StringIO(cur_str.replace(",", "\n")), header=None, names=[cur_col], skip_blank_lines=False)
                to_df.append(cur_df)
            df = pd.concat(to_df, axis=1, sort=False)
            if len(df) == 0:
                logger.warning("Length of timeseries dataframe was only 0 for visit_exam: " + str(visit_exam_id))
                error_message = 'Length of timeseries dataframe was only 0'
                df = pd.DataFrame()
            elif len(df) == 1:
                logger.warning("Length of timeseries dataframe was only 1 for visit_exam: " + str(visit_exam_id))
                error_message = 'Length of timeseries dataframe was only 1'
            else:
                df, cleaning_error = clean_df(df)
                if len(cleaning_error) > 0:
                    error_message = cleaning_error

        return df, error_message

    all_data = visit_exams_df['visit_exam_id'].apply(visit_exam_to_timeseries)
    visit_exams_df[['timeseries', 'error_timeseries']] = pd.DataFrame(all_data.to_list(), index=visit_exams_df.index)

    conn_ts.close()

    # filter out any empty dataframes
    # visit_exams_df = visit_exams_df[visit_exams_df['timeseries'].apply(lambda w: not w.empty)]

    # reset the index after these elements were deleted
    visit_exams_df = visit_exams_df.reset_index(drop=True)

    visit_exams_df.loc[:, 'blinks'] = visit_exams_df['timeseries'].apply(count_blinks)

    return visit_exams_df


def get_processed_path(subj_series):
    """
    get the path in S3 for a summary
    :return:
    """
    def remove_head_folder(w): # to remove the bucket name 'reactneuro-data' from the path
        return '/'.join(w.split('/')[1:])
    #           the folder,                        the name of the exam
    return remove_head_folder(subj_series['s3_folder']) + '/processed_' + subj_series['exam'].lower() + '.pickle'


def processed_missing(subj_series):
    """
    Check S3 to see if the summary output from analysis exists
    :param subj_series:
    :return:
    """
    path_s3 = get_processed_path(subj_series)
    return not s3_key_exists(path_s3)


def s3_key_exists(path_s3):
    s3_resource = boto3.resource('s3', **read_credentials())
    bucket = s3_resource.Bucket(BUCKET_DATA)
    objs = list(bucket.objects.filter(Prefix=path_s3))
    if any([w.key == path_s3 for w in objs]):
        return True
    else:
        return False




def overwrite_now(params, subj_series):
    """
    Check if the processed data file was modified recently, returns true if we need to overwrite the file
    This function is optimized to minimize the number of calls to S3 that are needed (though the code is
    a little bit more complex because of this (calls to processed_missing in different places).
    :param path_s3:
    :param overwrite:
              * True-> redo the analysis no matter what
              * False-> redo the analysis only if a file is missing
              * Integer-> redo the analysis if that many hours passed since the last processing
    :return: boolean
    """
    path_s3 = get_processed_path(subj_series)
    overwrite = False if 'overwrite' not in params else params['overwrite']
    if type(overwrite) == bool: # if its a boolean
        # if overwrite == True OR the file is missing
        return overwrite or processed_missing(subj_series)

    else: # check how long since the file was modified
        #  but first see if the file is missing
        if processed_missing(subj_series):
            return True

        else: # the file exists, so check the modify time:
            s3_client = boto3.client('s3', **read_credentials())
            response = s3_client.get_object(Bucket=BUCKET_DATA, Key=path_s3)
            utc_last_modified = response['LastModified']
            if type(overwrite) == int or type(overwrite) == float: # if its been a few days or not
                utc_now = datetime.datetime.now(datetime.timezone.utc)
                timedelta_since_last_modify = utc_now - utc_last_modified
                timedelta_overwrite_param = datetime.timedelta(days=overwrite)
                return timedelta_since_last_modify > timedelta_overwrite_param

            elif type(overwrite) == str and overwrite == 'batch': # the time must be a datetime
                if 'batch_begin_time' in params:
                    # if the last modify time was before we started doing this batch
                    return utc_last_modified < pd.Timestamp(params['batch_begin_time'])

                else: # the data must have been added after we started the batch processing
                    #   But at this stage we already know the file is not missing
                    return False
            else:
                raise Exception("Unrecognized batch processing type: " + str(overwrite) + " in params: " + str(params))

            #      if the timedelta since last modify is greater than the #hours set in the parameters



def load_processed_data(subj_series):
    path_s3 = get_processed_path(subj_series)

    s3_client = boto3.client('s3', **read_credentials())

    response = s3_client.get_object(Bucket=BUCKET_DATA, Key=path_s3)

    body = response['Body'].read()
    processed_data = pickle.loads(body)

    return processed_data


def load_s3_object(path_s3):
    s3_client = boto3.client('s3', **read_credentials())
    response = s3_client.get_object(Bucket=BUCKET_DATA, Key=path_s3)
    return response



def visit_exam_to_func(exam):
    """
#     get the analysis function for the specific exam
#     this could be from a lookup, or if we name things carefully then eval should work:

    :param exam:
    :return: None if the process function doesn't exist, otherwise returns handle for the function
    """
    func_str = 'process_' + exam.lower()
    if func_str in dir(analysis):
        func = eval("analysis." + func_str)
    else:
        func = None
    return func


def load_visit_exams_processed(visit_exams_df, params={}):
    """
    For every row - it will return summary stats for that visit_exam_id
    if a summary stat does not exist then:
        compute the summary using analysis.____
        save the summary json locally and on s3

    :param visit_exams_df:
    :param params:
    :return:
    """
    def visit_exam_to_processed(subj_series):
        exam = subj_series['exam'].lower()

        # check if we need to rerun analysis
        if overwrite_now(params, subj_series):
            complete_df_timeseries = load_visit_exams_timeseries(subj_series.to_frame().T)
            complete_df_audio = load_visit_exams_audio(subj_series.to_frame().T)
            audio_error = None
            timeseries_error = None
            analysis_error = None

            if len(complete_df_timeseries) == 0 and len(complete_df_audio) == 0:
                # todo: change every analysis script to return "no timeseries data" if it is missing
                processed_data = {'has_error': True,
                                  'error_description': 'no timeseries data and no audio data'}

            else:
                # check if your data is empty
                if len(complete_df_timeseries) == 0:
                    timeseries_df = None
                    timeseries_error = 'timeseries data empty'
                else:
                    timeseries_df = complete_df_timeseries.iloc[0]['timeseries']
                    timeseries_error = complete_df_timeseries.iloc[0]['error_timeseries']

                if  len(complete_df_audio) == 0:
                    audio_df = None
                else:
                    audio_df = complete_df_audio.iloc[0]['audio']
                    audio_error = complete_df_audio.iloc[0]['error_audio']

                #      choose which analysis function to run:
                func = visit_exam_to_func(exam)
                if func is None:
                    analysis_error = 'analysis.process_' + exam + 'function missing.'
                else:
                    # run the function
                    try:
                        processed_data = func(timeseries_df, audio_df)
                    except Exception as e:
                        analysis_error = 'process function failed with error: {}'.format(e)

                error_dict = {'timeseries_error': timeseries_error,
                              'audio_error': audio_error,
                              'analysis_error': analysis_error}

                if any(error is not None for error in error_dict.values()):
                    processed_data = {'has_error': True,
                                      'error_description': error_dict}

                visit_exam_id = subj_series['visit_exam_id']
                db_visit_exams_error_status_update(visit_exam_id, processed_data)


            # save the dictionary locally
        #     todo: get the 'path_head' functions in order over from report.py
        #           maybe save this in the same folder as figures, and then find and delete later if on local
            path_local = './reports/processed_data_tmp.pickle'
            with open(path_local, "wb") as f:
                pickle.dump(processed_data, f)

        # save the dictionary on s3
            # the place on s3 where it will be stored
            path_bucket = get_processed_path(subj_series)
            # upload it to s3 (and maybe delete?)
            dump_file_s3(path_local, path_bucket, bucket_name=BUCKET_DATA)

        else:
            processed_data = load_processed_data(subj_series)

        return processed_data

    # filter out any visit_exams that don't have complete data uploaded
    visit_exams_df = visit_exams_df[visit_exams_df['exam_version'].apply(lambda w: w is not None)]

    all_processed_data = visit_exams_df.apply(visit_exam_to_processed, axis=1)

    if len(all_processed_data) > 0:
        visit_exams_df = visit_exams_df.assign(processed=all_processed_data)

    return visit_exams_df

def db_visit_exams_error_status_update(visit_exam_id, processed_data):

    has_error = processed_data['has_error']
    if has_error:
        error_description = processed_data['error_description']
        # need to remove any single quotes from the error messages (this is required to insert this into SQL db)
        for error, message in error_description.items():
            if type(message) == str:
                error_description[error] = message.replace("'", "")
        error_description = json.dumps(error_description)
    else:
        error_description = ''
    query = """
            UPDATE dbo.visit_exams 
            SET error_description = '{0}', has_error = {1}
            WHERE id = '{2}' 
            """.format(error_description, has_error, int(visit_exam_id))
    run_execute(query)

    return query


def process_one_exam(visit_exams_df, ii=0):
    """
    Load the timeseries and process just a single visit_exam without try/catch
    for easy testing during development and catching errors

    :param visit_exams_df: regular visit exam df
    :param ii: index of the visit_exam you want to get
    :return:
    """
    subj_series = visit_exams_df.iloc[ii]
    complete_df_timeseries = load_visit_exams_timeseries(subj_series.to_frame().T)
    complete_df_audio = load_visit_exams_audio(subj_series.to_frame().T)

    timeseries_df = complete_df_timeseries.iloc[0]['timeseries']
    audio_df = complete_df_audio.iloc[0]['audio']

    exam = subj_series['exam']
    func = visit_exam_to_func(exam)
    processed_data = func(timeseries_df, audio_df)
    return processed_data


def human_readable_df(df):
    """
    If any of the numbered columns appear in the df ->
        look up the value of

    Note: error checking is in place, so we will catch it if
          ever an ID is inserted into data table database but not to
          the corresponding lookup table

    :param df: any df at all, totally safe to try this out
    :return: a dataframe with new human-readable columns

    It converts the following columns to their real names:
       exam_id, location_id	device_id

    """
    key_to_table = {'exam_id': 'exams',
                    'location_id': 'locations',
                    'device_id': 'devices'}

    for key, table in key_to_table.items():
        if key in df:
            name_col = key.replace("_id", "")

            if name_col not in df:
                mapper = etl.reverse_dict(pd_get_id_map(table))
                # n_missing = np.sum(np.isnan(df['device_id']))
                n_missing = np.sum(pd.isnull(df['device_id']))
                if n_missing > 0:
                    logger.warning(table + ":" + key + " contains " + str(n_missing) + " NaN values")
                try:
                    # replace the number with its type, but if it is nan then don't warn here
                    names = df[key].apply(lambda w: mapper[w] if not pd.isnull(w) else 'NaN')

                except KeyError:
                    # failed to map between names and ids
                    # lets keep going but note this in the logs
                    logger.warning("Key missing from the " + key + " column")
                    print(mapper)
                else: # our names were created OK
                    df.loc[:, name_col] = names

    return df


def subj_series_to_complete_df(subj_series):
    """
    get the complete_df from a subject series
    :param subj_series:
    :return:
    """
    visit_id = subj_series['visit_id']
    visit_exam_id = subj_series['visit_exam_id']
    cur_visits = get_visits(visit_id=visit_id)
    visit_exams_df = get_visit_exams(cur_visits, id=visit_exam_id)
    complete_df = load_visit_exams_timeseries(visit_exams_df)
    return complete_df


def get_recent_exams(n=10, exam='none'):
    """
    Get the recent exams of a certain kind
    :param n: number of *visits you want to look for (might not return the right number if those visits didn't try your
              exam, use np.inf to get all visits for that exam
    :param exam: the exam you are looking for, leave blank to get all exams
    :return: complete DF including the timeseries
    """
    visit_df = get_visits(n=n)
    visit_exams_df = get_visit_exams(visit_df)
    if exam != 'none':
        filter_criteria = lambda w: w['exam'] == exam
        to_load = visit_exams_df[filter_criteria(visit_exams_df)]
    else:
        to_load = visit_exams_df
    # complete_df = load_visit_exams_timeseries(to_load)
    complete_df = to_load
    return complete_df


def get_one_recent_exam():
    visit_exams_df = get_visit_exams(get_visits(n=np.inf))
    # remove any without data uplaoded
    visit_exams_df = visit_exams_df[visit_exams_df['exam_version'].apply(lambda w: w is not None)]
    # pick the most recent one
    recent_exams = visit_exams_df.loc[visit_exams_df.groupby('exam')['created_date'].idxmax()]
    # return just those exams
    return get_visit_exams_by_id(recent_exams['visit_exam_id'].values)


def processed_status_good(w):
    """
    check if the results dataframe has good results in it
    :param w: a
    :return:
    """

    if type(w) == dict:
        if 'status' in w:
            return w['status'] == True
        else:
            return True
    else: # if its a dataframe or list, make sure something is in there
        return len(w) > 0


def split_current_control_df(df_or_complete_df, visit_exam_id, longitudinal=False):
    """
    Separate the current and control dataframes
    :param df_or_complete_df:
    :param visit_exam_id: if the visit_exam_id is `None` then current_df is df_or_complete_df
    :return:
    """

    if visit_exam_id is None:
        current_df = df_or_complete_df
        control_df = None
    else:
        if longitudinal:
            # get any exams by that user
            current_df = df_or_complete_df.loc[df_or_complete_df['visit_exam_id'] == visit_exam_id]
            cur_user = current_df.iloc[0]['subject_id']
            current_df = df_or_complete_df.loc[df_or_complete_df['subject_id'] == cur_user]

            control_df = df_or_complete_df.loc[df_or_complete_df['visit_exam_id'] != visit_exam_id]
        else:
            # just split
            current_df = df_or_complete_df.loc[df_or_complete_df['visit_exam_id'] == visit_exam_id]
            control_df = df_or_complete_df.loc[df_or_complete_df['visit_exam_id'] != visit_exam_id]

    # filter out any control data that is bad
    control_df = control_df[control_df['processed'].apply(lambda w: not w['has_error'])]
    if len(control_df) == 0:
        control_df = None

    return current_df, control_df


def load_json(cur_file):
    with open(cur_file) as json_file:
        results = json.load(json_file)
    return results

def get_host(config_file = CONFIG_FILE):
    db = load_json(config_file)
    host_to_use = db['db']
    return host_to_use

def set_host(host, config_file = CONFIG_FILE):
    """
    Change the host in the .aws/config file
    :param host: 'staging' or 'production'
    :param config_file: defaults to '.aws/production_staging_config.json'
    :return: None
    """
    if host not in {'production', 'staging'}:
        raise Exception("Invalid host: " + str(host))
    else:
        db = {
                "db": host
            }
        etl.dump_json(db, config_file)


def get_conn(config_file = CONFIG_FILE):
    host_to_use = get_host(config_file)
    if host_to_use == 'production':
        config = load_json('.aws/production_db.json')
    elif host_to_use == 'staging':
        config = load_json('.aws/staging_db.json')
    else:
        raise ValueError('DB name not valid.')

    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        return conn

    except Exception as e:
        print("DB connection failed with exception:")
        print(e)
        return None

def run_execute(query, *varargs):
    conn = get_conn()
    cursor = conn.cursor()
    if len(varargs) == 0:
        cursor.execute(query)
    elif len(varargs) == 1:
        cursor.execute(query, (varargs[0],))
    else:
        logger.error("too many arguments passed to run_execute")
    cursor.close()
    conn.commit()
    conn.close()
    return conn


def read_credentials(loc=".aws/credentials"):
    """
    Read AWS credentials for access to S3
    :param loc: location of credentials file
    :return:
    """
    with open(loc,"r") as f:
        lines=f.readlines()
    config = {'aws_access_key_id': lines[1].split("=")[1].strip(),
               'aws_secret_access_key': lines[2].split("=")[1].strip()}
    return config





def dump_file_s3(path_local, path_bucket, bucket_name=BUCKET_DATA):
    """
    Different options for doing this
    :param visit_id:
    :param fpath:
    :param object:
    :return:
    """
    logger.debug("Uploading file to s3: " + path_local + " " + path_bucket)
    s3_resource = boto3.resource('s3', **read_credentials())
    s3_resource.Bucket(bucket_name).upload_file(path_local, path_bucket)
    return True


def stream_file_s3(path_s3, bucket_name=BUCKET_DATA):
    """
    stream the file from S3 into a string object
    """

    s3_resource = boto3.resource('s3', **read_credentials())
    bucket = s3_resource.Bucket(bucket_name)
    objs = list(bucket.objects.filter(Prefix=path_s3))
    # look for exact match
    objs = [w for w in objs if w.key == path_s3]
    if len(objs) == 1:
        return objs[0].get()['Body'].read().decode('utf-8')
    else:
        raise Exception("Tried to read missing key: " + path_s3)


def load_audio(uri):
    """
    Function to lad audio from s3 object
    :param uri:
    :return:
    """
    # print(' ... processing:', uri)

    bucket = 'reactneuro-data'
    filepath = '/'.join(uri.split('/')[1:])

    # LOAD OBJECT
    s3 = boto3.client('s3', **read_credentials())
    obj = s3.get_object(Bucket=bucket, Key=filepath)
    # print(' ... object loaded')

    # UNWRAP OBJECT
    data = io.BytesIO(obj['Body'].read())

    # READ AUDIO
    audio = AudioSegment.from_file(data)
    # print(' ... audio read')

    return audio

def get_transcription_amazon(job_name, job_uri):
    """
    Transcribes an audio file using Amazon Transcribe
    :param job_name: an arbitrary name that amazon job manager uses --> cannot be duplicated
    :param job_uri: location of file on s3
    :return: transcript_df
    """

    # open Amazon Transcribe client and start transcription job
    transcribe_client = boto3.client('transcribe', **read_credentials(), region_name='us-east-1')
    try:
        transcribe_client.start_transcription_job(TranscriptionJobName=job_name,
                                                  Media={'MediaFileUri': job_uri},
                                                  MediaFormat='wav',
                                                  LanguageCode='en-US')

        # get the status of the transcription while it is occurring and wait for it to finish
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(20)

        logger.info('Transcription finished')

        # Once transcription is complete, pull out the transcript from the amazon URL
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
            transcript = json.loads(response.read())
        else:
            transcript = None
            raise Exception('Transcription failed')

        # convert transcript output to a df so we can use it downstream
        if (transcript is not None) and (len(transcript['results']['items']) > 0):
            transcript_df = pd.DataFrame.from_dict(transcript['results']['items'])
            transcript_df.dropna(subset=['start_time'], inplace=True)
            transcript_df.drop(columns=['type'], inplace=True)
            transcript_df['transcript'] = transcript_df['alternatives'].apply(lambda w: w[0]['content'])
            transcript_df.dropna(subset=['transcript'], inplace=True)
        else:
            transcript_df = None

        # delete the job from the amazon transcription job queue
        delete_job(job_name, transcribe_client)

    except Exception as e:
        logger.warning('Transcription Process not started for job: {0} with exception: {1}'.format(job_uri, e))
        transcript_df = None

    return transcript_df

def delete_job(job_name, transcribe_client):
    try:
        transcribe_client.delete_transcription_job(
            TranscriptionJobName=job_name)
        logger.info("Deleted job %s.", job_name)
    except ClientError:
        logger.exception("Couldn't delete job %s.", job_name)
        raise

def load_audio_parselmouth(uri):
    """
    Function to load parselmouth object
    :param uri:
    :return:
    """
    # load audio
    audio = LoadAudio(uri)

    # export it to a temporary file
    file_handle = audio.export("tmp.wav", format="wav")

    # read file
    with io.open("tmp.wav", "rb") as f:
        content = f.read()

    # import as parselmouth object
    snd = parselmouth.Sound("tmp.wav")

    # remove file
    os.remove("tmp.wav")

    return snd


def get_exam_uri(exam_name):
    """
    Function to retrieve all filepaths to audio for specified exam
    :param exam_name:
    :return:
    """
    # DICT THAT MAPS EXAM ID TO NAME
    # todo: check if these examName2Id is still accurate. I think it's actually CF=18 and Stroop=17
    examName2Id = {'CategoryFluency': 11, 'Stroop': 12}
    examName2csvPrefix = {'CategoryFluency': 'CategoryFluency--v0.1--',
                          'Stroop': {0: 'Stroop--v0.1--',
                                     1: 'Stroop--v0.3--',
                                     2: 'Stroop--v1.0--',
                                     3: 'Stroop--v1.1--',
                                     # 4:'Stroop--v#0.4-1.2--',
                                     }}

    # GET EXAM ID
    exam_id = examName2Id[exam_name]

    # INIT DICT TO STORE RESULTS
    exam2uri = {}

    # FIND VISIT_ID THAT HAD THE EXAM OF INTEREST ADMINISTERED
    df = pd_read_sql_conn(
        "SELECT * FROM dbo.{table} WHERE exam_id = {exam_id}".format(table='visit_exams', exam_id=exam_id))

    # GET LIST OF VISIT EXAM IDS
    visit_exam_ids = df.id

    # LOOP THROUGH EACH EXAM ID
    # for id in visit_exam_ids:
    for _, row in df.iterrows():

        # print('row', row)
        id = row.id
        user_id = row.visit_id

        # FIND FILEPATH OF VISIT_EXAM_ID FOR THE TEST WE ARE INTERESTED IN
        df_files = pd_read_sql_conn(
            "SELECT * FROM dbo.{table} WHERE visit_exam_id = {visit_exam_id}".format(table='files', visit_exam_id=id))
        # print(df_files)
        # PRINT FILEPATH
        if df_files.uri.empty:
            continue

        else:

            # init
            exam2uri[id] = {}
            exam2uri[id]['user_id'] = user_id

            # ADD AUDIO FILE PATHS
            uri = df_files.uri.tolist()

            fix_uri_list = []
            for u in uri:
                if 'reactneuro-data' not in u.split('/')[0]:
                    fix_uri_list.append('reactneuro-data/' + u)
                else:
                    fix_uri_list.append(u)

            # exam2uri[id]['uri'] = uri
            exam2uri[id]['uri'] = fix_uri_list

            # ADD CSV FILEPATHS
            # user_id = exam2uri[id]['uri'].split('/')[-3]
            user_dir = '/'.join(uri[0].split('/')[:-2])

            if exam_name == 'Stroop':
                csvpaths = []
                for i in range(len(examName2csvPrefix[exam_name])):
                    # print(i, type(i), examName2csvPrefix[exam_name])
                    csvpath = 's3://' + user_dir + '/' + examName2csvPrefix[exam_name][i] + user_id + '.csv'
                    csvpaths.append(csvpath)
                exam2uri[id]['csv'] = csvpaths

            else:
                csvpath1 = 's3://' + user_dir + '/' + examName2csvPrefix[exam_name] + user_id + '.csv'
                # csvpath2 = 's3://' + user_dir + '/' + examName2csvPrefix[exam_name][2] + user_id + '.csv'
                exam2uri[id]['csv'] = [csvpath1]

    return exam2uri

def get_complete_df_for_user(subject_id, **kargs):
    if subject_id is None:
        # if you didn't pass a visit ID just get the most recent user
        kargs = {'n': 1}
    else:
        kargs = {'subject_id': subject_id, 'n': np.inf}

    visits = get_visits(**kargs)
    visit_exams_df = get_visit_exams(visits)
    complete_df = load_visit_exams_timeseries(visit_exams_df)
    return complete_df

def audio_verification(subj_series):
    s3_path = subj_series['s3_folder']
    exam = subj_series['exam']
    fullpath = s3_path + '/' + exam + '/'
    prefix = fullpath.replace(BUCKET_DATA + "/", "")

    s3_client = boto3.client('s3', **read_credentials())
    response = s3_client.list_objects_v2(Bucket=BUCKET_DATA, Prefix=prefix)
    audio_exist = False
    file_size_mb = 0

    if 'Contents' in response:
        for key in response['Contents']:
            if '.wav' in key['Key']:
                audio_exist = True
                file_size_mb = key['Size']/1000000
            else:
                continue
    return audio_exist, file_size_mb


def csv_verification(subj_series):
    s3_path = subj_series['s3_folder']
    prefix = s3_path.replace(BUCKET_DATA + "/", "")

    s3_resource = boto3.resource('s3', **read_credentials())
    bucket = s3_resource.Bucket(BUCKET_DATA)

    keys = []
    for file in bucket.objects.filter(Prefix=prefix):
        keys.append(file.key)

    exam = subj_series['exam']
    exam_version = subj_series['exam_version']
    visit_id = subj_series['visit_id']

    csv_exist = False
    file_size_mb = 0

    if all([w is not None for w in [exam, exam_version, visit_id]]):
        csv_file = exam + '--' + exam_version + '--' + visit_id + '.csv'
        keypath = prefix + '/' + csv_file
        if any(csv_file in key for key in keys):
            csv_exist = True
            file = s3_resource.Object(BUCKET_DATA,keypath)
            file_size = file.content_length
            file_size_mb = file_size/1000000

    return csv_exist, file_size_mb



def get_test_subjects():
    """
    Get all the ids of the test subjects from both:
        * named "test" as a first name
        * anyone who has ever been tested at a test site

    :return: set of all subject ids who are in test
    """

    query = """
    SELECT id, first_name FROM dbo.{};
    """.format('subjects')
    df = pd_read_sql_conn(query)
    test_subjs = set(df[df['first_name'].apply(lambda w: 'test' in w.lower())]['id'].values)


    cur_map = pd_get_id_map('locations')
    test_location_ids = set([cur_map[w] for w in TEST_LOCATION_NAMES if w in cur_map])

    query = """
            SELECT id, subject_id, location_id 
            FROM dbo.visits
        """
    df = pd_read_sql_conn(query)

    test_subjs |= set(df[df['location_id'].apply(lambda w: w in test_location_ids)]['subject_id'].values)

    return test_subjs



"""
todo: 
    when this becomes a class, the __init__ function should load 
    the connection and then pull all those mappings into a dict
"""
# load in some useful mappings
column2id = pd_get_id_map('columns')
exam2id = pd_get_id_map('exams')
id2column = etl.reverse_dict(column2id)
id2exam = etl.reverse_dict(exam2id)
