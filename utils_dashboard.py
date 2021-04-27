import utils_db as udb
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from matviz.matviz import etl

def get_progress_summary(timeframe, drop_react=True, drop_test=False):
    """
    Gets all of the visits and visit_exams during the timeframe specified in days
    :param timeframe: number of days prior to current_date --> e.g. timeframe = 7 is the last week
    :return: summary_df
    """
    if len(timeframe) == 1:
        if timeframe[0]=='CURRENT_DATE':
            date_str = "WHERE visits.created_date > " + timeframe[0]
        else:
            date_str = "WHERE visits.created_date > '" + timeframe[0] + "'"
    elif len(timeframe) == 2:
        date_str = "WHERE visits.created_date BETWEEN '" + timeframe[0] + " 00:00:00' AND '" + timeframe[1] + " 23:59:59'"
    else:
        raise Exception("please input a valid date range, timeframe can only be 1 or 2 values")
    query = """
                 SELECT visits.id as visit_id, 
                        dbo.visit_exams.id as visit_exam_id,
                        dbo.visit_exams.exam_id,
                        dbo.visit_exams.exam_version,
                        visits.created_date,
                        visits.reports_processed,
                        dbo.visit_exams.has_error,
                        dbo.visit_exams.error_description,
                        visits.s3_folder,
                        visits.updated_at,
                        visits.location_id,
                        visits.device_id,
                        dbo.subjects.first_name,
                        dbo.subjects.id as subject_id
                    from dbo.visits as visits
                LEFT JOIN dbo.visit_exams ON visits.id=dbo.visit_exams.visit_id
                RIGHT JOIN dbo.subjects ON visits.subject_id=dbo.subjects.id
                {}
                ORDER BY updated_at DESC
    """.format(date_str, timeframe)
    conn = udb.get_conn()
    summary_df = pd.read_sql(query, conn)
    exam_mapper = etl.reverse_dict(udb.pd_get_id_map('exams'))
    summary_df['exam'] = summary_df['exam_id'].apply(lambda w: exam_mapper[w] if not pd.isnull(w) else 'NaN')
    location_mapper = etl.reverse_dict(udb.pd_get_id_map('locations'))
    summary_df['location'] = summary_df['location_id'].apply(lambda w: location_mapper[w] if not pd.isnull(w) else 'NaN')

    if drop_react == True:
        summary_df = summary_df.loc[~(summary_df['location'] == 'Test Site one')]

    summary_df['subject'] = np.where(summary_df['first_name']=='Test', 'Test', 'User')
    summary_df.drop(['first_name'], axis=1, inplace=True)

    if drop_test == True:
        summary_df = summary_df.loc[~(summary_df['subject'] == 'Test')]

    # audio_present = summary_df.apply(udb.audio_verification, axis=1)
    # csv_present = summary_df.apply(udb.csv_verification, axis=1)
    # summary_df.assign(audio_present=audio_present, data_present=csv_present)

    return summary_df




def get_quick_view_progress(timeframe,  drop_test, include_orphaned=True, drop_react=True):
    """
    aggregates the visit_exams by visit_id and displays whether processing was successful
    :param timeframe:
    :param include_orphaned:
    :return:
    """
    summary_df = get_progress_summary(timeframe, drop_react, drop_test)
    created_at_df = summary_df[['visit_id', 'created_date']]
    location_df = summary_df[['visit_id', 'location']]
    device_df = summary_df[['visit_id', 'device_id']]
    subject_df = summary_df[['visit_id', 'subject']]

    # making new columns that allow us to assess which ids are having problems
    summary_df['processing_attempted'] = summary_df['has_error'].replace({True: 1, False: 1})
    summary_df['processing_failed'] = summary_df['has_error'].replace({True: 1, False: 0})
    summary_df['data_uploaded'] = summary_df['exam_version'].notnull().replace({True: 1, False: 0})
    summary_df['visit_exam_count'] = summary_df['visit_exam_id'].notnull().replace({True: 1, False: 0})

    if include_orphaned == False:
        summary_df = summary_df.loc[summary_df['visit_exam_id'].notnull()]

    # grouping by visit_id so we can visualize more easily
    quick_view = summary_df.groupby(by=["visit_id"], sort=False).sum(skipna=False)
    quick_view['num_unprocessed'] = quick_view['data_uploaded'] - quick_view['processing_attempted']
    quick_view['need_to_check'] = (quick_view['processing_failed'] > 0) | (quick_view['num_unprocessed'] != 0)
    quick_view['missing_visit_exams'] = (quick_view['visit_exam_count'] == 0)
    quick_view['data_missing'] = (quick_view['visit_exam_count'] > quick_view['data_uploaded'])

    # casting each column to int just so it's easier on the retinas
    quick_view[['data_uploaded',
             'processing_attempted',
             'processing_failed',
             'num_unprocessed']] = quick_view[['data_uploaded',
                                            'processing_attempted',
                                            'processing_failed',
                                            'num_unprocessed']].astype(int)

    # keeping only the useful columns for this quick view
    quick_view = quick_view[['visit_exam_count',
                             'data_uploaded',
                             'processing_attempted',
                             'processing_failed',
                             'num_unprocessed',
                             'need_to_check',
                             'data_missing',
                             'missing_visit_exams']]

    # adding in the created_date back in and resetting the index a few times to deal with pandas v0.25.3
    quick_view = quick_view.join(created_at_df.set_index('visit_id'), on='visit_id')
    quick_view = quick_view.join(location_df.set_index('visit_id'), on='visit_id')
    quick_view = quick_view.join(device_df.set_index('visit_id'), on='visit_id')
    quick_view = quick_view.join(subject_df.set_index('visit_id'), on='visit_id')
    quick_view = quick_view.reset_index()
    quick_view = quick_view.drop_duplicates(subset=['visit_id'], keep='first')
    quick_view = quick_view.reset_index(drop=True)

    quick_view['datetime_EST'] = pd.to_datetime(quick_view['created_date']) - timedelta(hours=5)
    quick_view.drop(['created_date'], axis=1, inplace=True)

    return quick_view, summary_df



def diagnostic_tools(timeframe, drop_test, drop_react=True):
    """
    displays number of visits and visit_exams and adds up the successes and failures. makes a helpful chart as well
    :param timeframe:
    :return:
    """
    summary_df = get_quick_view_progress(timeframe, drop_react, drop_test)

    summary_df['success'] = (summary_df['need_to_check'] == False) & (summary_df['processing_attempted'] > 0)
    diagnostic_dict = dict()
    diagnostic_dict['num_visits'] = summary_df.shape[0]
    diagnostic_dict['num_visit_exams_attempted'] = summary_df['visit_exam_count'].sum()
    diagnostic_dict['num_visits_w_full_data'] = len(
        summary_df.loc[(summary_df['data_missing'] == False) & (summary_df['data_uploaded'] > 0)])
    diagnostic_dict['num_visits_data_missing'] = summary_df['data_missing'].sum()

    diagnostic_dict['num_processing_attempted'] = summary_df['processing_attempted'].sum()
    diagnostic_dict['num_processing_failed'] = summary_df['processing_failed'].sum()
    diagnostic_dict['num_unprocessed'] = summary_df.loc[summary_df['data_missing'] == False]['num_unprocessed'].sum()

    diagnostic_df = pd.DataFrame.from_dict(diagnostic_dict, orient='index')

    summary_df['time_of_creation_EST'] = summary_df['datetime_EST'].dt.time

    uch.plot_visit_id_creation(summary_df)

    return diagnostic_df, summary_df

def get_failed_visit_exams(timeframe):
    df = get_progress_summary(timeframe)
    failed_visit_exams = df.loc[(pd.isnull(df['exam_version'])) | (df['has_error'] == True)]

    csv_data = failed_visit_exams.apply(udb.csv_verification, axis=1)
    failed_visit_exams[['csv_on_s3', 'csv_size_mb']] = pd.DataFrame(csv_data.to_list(), index=failed_visit_exams.index)

    audio_data = failed_visit_exams.apply(udb.audio_verification, axis=1)
    failed_visit_exams[['audio_on_s3', 'wav_size_mb']] = pd.DataFrame(audio_data.to_list(), index=failed_visit_exams.index)

    failed_visit_exams['datetime_EST'] = pd.to_datetime(failed_visit_exams['created_date']) - timedelta(hours=5)

    failed_visit_exams = failed_visit_exams.drop(columns=['exam_id', 'location_id', 'updated_at', 'created_date'])
    failed_visit_exams = failed_visit_exams.reset_index(drop=True)

    grouped_failed = failed_visit_exams.groupby(by=['visit_id'], sort=False)

    html = ''
    for visit, exams in grouped_failed:
        html += exams.to_html()

    return failed_visit_exams, html


def get_visits_by_location(timeframe):
    date_str = "WHERE visits.created_date > '" + timeframe[0] + "'"
    query = """
         SELECT visits.id as visit_id, 
                dbo.visit_exams.id as visit_exam_id,
                dbo.visit_exams.exam_id,
                dbo.visit_exams.exam_version,
                visits.created_date,
                visits.updated_at,
                visits.location_id,
                visits.device_id,
                dbo.subjects.first_name,
                dbo.subjects.last_name,
                dbo.subjects.id as subject_id
            from dbo.visits as visits
        LEFT JOIN dbo.visit_exams ON visits.id=dbo.visit_exams.visit_id
        RIGHT JOIN dbo.subjects ON visits.subject_id=dbo.subjects.id
        {}
        ORDER BY updated_at DESC
    """.format(date_str, timeframe)
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)

    location_mapper = etl.reverse_dict(udb.pd_get_id_map('locations'))
    df['location'] = df['location_id'].apply(lambda w: location_mapper[w] if not pd.isnull(w) else 'NaN')
    df['month'] = df['created_date'].dt.month
    df['year'] = df['created_date'].dt.year
    df['date'] = df['created_date'].dt.date
    df = df[['visit_id', 'created_date', 'updated_at', 'location', 'month', 'year', 'date']]

    visits_by_location = df.groupby(['location', 'month', 'year']).count()
    vbl = visits_by_location.reset_index()
    vbl = vbl.drop(['created_date', 'updated_at', 'date'], axis=1)
    vbl = vbl.rename(columns={'visit_id': 'visits'})

    month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                  5: 'May', 6: 'June', 7: 'July', 8: 'August',
                  9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    vbl['month'] = vbl['month'].apply(lambda w: month_dict[w] if w in month_dict.keys() else w)

    return df, vbl


def get_subjects_table():
    query = """
             SELECT subjects.first_name,
                    subjects.last_name,
                    subjects.id
                from dbo.subjects as subjects
            ORDER BY updated_at DESC
            """
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)

    df['full_name'] = subjects_df['first_name'] + ' ' + subjects_df['last_name']
    subjects_lookup_table = dict(zip(df['id'], df['full_name']))

    return subjects_lookup_table


def get_visits_by_subject_by_month():
    query = """
         SELECT visits.id as visit_id, 
                visits.created_date,
                visits.updated_at,
                visits.location_id,
                visits.device_id,
                visits.subject_id
            from dbo.visits as visits

        ORDER BY updated_at DESC
    """
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)

    location_mapper = etl.reverse_dict(udb.pd_get_id_map('locations'))
    df['location'] = df['location_id'].apply(lambda w: location_mapper[w] if not pd.isnull(w) else 'NaN')
    df['month'] = df['created_date'].dt.month
    df['year'] = df['created_date'].dt.year
    df['date'] = df['created_date'].dt.date

    subjects_table = get_subjects_table()

    subjects_grouped = df.groupby(['location', 'subject_id', 'month', 'year']).count()
    subjects = subjects_grouped.reset_index()
    subjects['subject'] = subjects['subject_id'].apply(lambda w: subjects_table[w] if w in subjects_table else w)

    subjects_visits = subjects[['location', 'month', 'year', 'subject', 'visit_id']]
    subjects_visits = subjects_visits.rename(columns={'visit_id': 'visits'})

    return subjects_visits