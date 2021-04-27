# %load_ext autoreload
# %autoreload 2

import sys
import os
import datetime
import time

import numpy as np
import pandas as pd
from matviz.matviz import etl

import chart as chart
import utils_db as udb
import utils_df as udf
import utils_git as ugit
import utils_controls as ucon
import jinja2
import json
import requests
import pickle

import traceback

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)

CONTROL_SUBJECT_QTY = 50
# CONTROL_SUBJECT_QTY = 3
STAGING_POST_URL = "https://reactneuro-portal-staging.herokuapp.com/api/v1/"
PRODUCTION_POST_URL = 'https://reactneuro-portal.herokuapp.com/api/v1/'


EXAM_TO_VIZ_FUNC = {'prosaccade': chart.plot_prosaccade,
                       'convergence': chart.plot_convergence,
                       'pupillaryreflex': chart.plot_pupillaryreflex,
                       'gaze': chart.plot_gaze,
                       'smoothpursuit': chart.plot_smoothpursuit,
                       'smoothpursuit2d': chart.plot_smoothpursuit2d,
                       'trailmaking': chart.plot_trailmaking,
                       'trailmaking2': chart.plot_trailmaking,
                       'selfpacedsaccade': chart.plot_selfpacedsaccade,
                       'categoryfluency': chart.plot_categoryfluency,
                       'letterfluency': chart.plot_letterfluency,
                       'stroop': chart.plot_stroop,
                       'bostonnaming': chart.plot_bostonnaming,
                       'digitspanforward': chart.plot_digitspanforward,
                       'digitspanbackward': chart.plot_digitspanbackward,
                       'tapping': chart.plot_tapping,
                       'memoryencoding': chart.plot_memoryencoding,
                       'memoryrecall': chart.plot_memoryrecall
                       }

def visit_exam_to_func(exam):
#     get the analysis function for the specific exam
#     this could be from a lookup, or if we name things carefully then eval should work:
    func = eval('chart.plot_' + exam.lower())
    return func


def get_processed_df_for_visit(visit_id, params={}):
    visit_exams_df = udb.get_visit_exams(udb.get_visits(visit_id=visit_id))
    processed_df = udb.load_visit_exams_processed(visit_exams_df, params)
    return processed_df


def concatenate_history(subj_series):
    """
    Get all the visit_exams for a specific user & exam combination
    :param subj_series:
    :return:
    """

    # todo: check get_visits is working with user_id here
    visits = udb.get_visits(subject_id=subj_series['subject_id'])

    visit_exams_df = udb.get_visit_exams(visits)
    visit_exams_df.dropna(subset=['exam'], inplace=True)

    filter_exam = visit_exams_df.exam == subj_series['exam']

    visit_exams_df = visit_exams_df[filter_exam]

    return visit_exams_df



def concatenate_control(subj_series, n=CONTROL_SUBJECT_QTY, year_range=20, max_controls = False):
    """
    :param subj_series: the series for the visit_exam we are looking to get controls for
    :param n: maximum number of controls
    :param year_range: +- year range around which control subjects will be taken
    :param max_controls: True if there are no restrictions on who is a control - except that it not be the subject
    :return:
    """
    visits = udb.get_visits(n=np.inf)
    # todo: test getting a single exam in the query
    visit_exams_df = udb.get_visit_exams(visits)

    visit_exams_df.dropna(subset=['exam'], inplace=True)
    visit_exams_df.dropna(subset=['exam_version'], inplace=True)

    # just the visit_exams with our exam
    filter_exam = visit_exams_df.exam == subj_series.exam
    # just the visit_exams *not including any from our subject
    filter_username = np.invert(visit_exams_df.subject_id == subj_series.subject_id)
    # do the filter step
    visit_exams_df = visit_exams_df[filter_exam & filter_username]

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

    control_and_subject_df = control_df.append(subj_series)

    return control_and_subject_df


def check_params_for_exam(subj_series, params):
    """
    Parse if this exam needs to be processed or not
    used in chart_and_save and replay_and_save
    """
    if 'exams' in params:

        exams = params['exams']

        # check if you passed in a string when exams should be a list
        if type(exams ) == str:
            exams = [exams]

        # normalize exam list into lowercase
        exams = [w.lower() for w in exams]

        if subj_series['exam'].lower() in exams:
            return True
        else:
            return False
    else:
        return True

def get_path_local(func, subj_series):
    visit_id = subj_series['visit_id']
    visit_exam_id = subj_series['visit_exam_id']
    path_head = get_path_head(visit_id, visit_exam_id)

    # extension is always png for figures
    ext = 'png'
    path_suffix = func.__name__ + '.' + ext
    path_local = path_head + path_suffix

    return path_local

# todo: WHERE IS THIS USED?
# todo: see if we can figure out a way to actually use this function (besides in the test)
def get_path_s3(func, subj_series):
    path_local = get_path_local(func, subj_series)
    host = udb.get_host()
    if host == 'production':
        path_bucket = path_local.replace(udf.PATH_REPORTS, udb.PATH_AWS_PRODUCTION)
    elif host == 'staging':
        path_bucket = path_local.replace(udf.PATH_REPORTS, udb.PATH_AWS_STAGING)
    else:
        logger.error('Host name not recognized. Please check config file')

    return path_bucket


def path_local_to_s3(cur_path):
    host = udb.get_host()
    if host == 'production':
        path_s3 = udb.PATH_AWS_PRODUCTION
    elif host == 'staging':
        path_s3 = udb.PATH_AWS_STAGING
    else:
        logger.error('Host name not recognized. Please check config file')

    return cur_path.replace(udf.PATH_REPORTS, path_s3)

def chart_and_save(func, processed_df, params):
    """
    Chart the graph, and save the file,
    Then if the host is 'aws' upload it to
    :param func: the charting function
    :param path_head: the path where the file should be saved (local)
    :param host: aws or local
    :param varargin: the data used for charting (1 or more vars)
    :return: the local path
    """
    subj_series = processed_df.iloc[-1]
    visit_exam_id = subj_series['visit_exam_id']
    visit_id = subj_series['visit_id']
    exam = subj_series['exam']

    path_local = get_path_local(func, subj_series)

    # run the charting function if we want to!
    if check_params_for_exam(subj_series, params):
        # make the chart 'func' applied to visit_exam_id with the data in summary_df
        # only if the analysis has no errors
        if 'has_error' in subj_series['processed']:
            if subj_series['processed']['has_error'] == False:
                try:
                    status = func(processed_df, visit_exam_id)
                    # todo: save this status in the database as json
                except Exception as e:
                    logger.warning('Could not chart - {} with error:'.format(subj_series['exam']))
                    logger.warning(str(e))
                    logger.warning(traceback.format_exc())
            else:
                print("skipped charting for: " + str(subj_series['visit_exam_id']) + str(subj_series['processed']))
        else:
            print("no has_error found in processed column")
            Warning("please export a has_error field in your analysis results dictionary for: " + subj_series['exam'])

        # save the chart locally
        chart.save_and_close(path_local)

        # upload to S3 (if needed also delete local)
        dump_chart_s3(path_local, params['host'])

    return path_local


def dump_chart_s3(path_local, host):
    """
    Saves a file on S3 and also locally depending on host parameter
    :param path_local:
    :param host: can be 'aws', 'local' or 'local aws'
    :return:
    """
    if 'aws' in host:
        # replace './reports' with 'production'
        path_to_use = udb.get_host()
        if path_to_use == 'production':
            path_bucket = path_local.replace(udf.PATH_REPORTS, udb.PATH_AWS_PRODUCTION)
        elif path_to_use == 'staging':
            path_bucket = path_local.replace(udf.PATH_REPORTS, udb.PATH_AWS_STAGING)
        udb.dump_file_s3(path_local, path_bucket)

    if 'local' not in host:
        os.remove(path_local)

    if 'local' not in host and 'aws' not in host:
        logger.error("Invalid host name: " + str(host) + ". Only ['aws', 'local', 'aws local] are valid.")



def replay_and_save(subj_series, params):
    """
    Create the mp4 replay and upload it if we need to
    :return: the local path
    """

    path_local = get_path_head(subj_series['visit_id'], subj_series['visit_exam_id']) + 'replay.mp4'

    # plot the videos if it is empty OR if the parameter is set to true
    if 'videos' not in params or params['videos']:
        # check also that we are interested in running this exam              also check the lowercase version plz
        if check_params_for_exam(subj_series, params):
            # run the charting function!
            chart.replay_digital_exam(subj_series, path_local)
            # dump it to s3 if we need to
            dump_chart_s3(path_local, params['host'])
        else:
            logger.debug("Skipping charts for: " + subj_series['exam'])

    return path_local


def local_path_to_html_path(path_local):
    # get rid of the './reports'
    path_tmp = path_local.replace(udf.PATH_REPORTS, "")
    # get rid of the visit ID folder part, without knowing what it is
    return os.path.join(*(path_tmp.split(os.path.sep)[2:]))


def get_dir_visit(visit_id):
    # get the local directory for this visit
    path_dir = os.path.join(udf.PATH_REPORTS, visit_id)
    etl.robust_mkdir(path_dir)
    return path_dir

def get_path_head(visit_id, visit_exam_id):
    # get the local path for the figures (folder and start of the filename)
    dir_visit = get_dir_visit(visit_id)
    dir_figs = os.path.join(dir_visit, 'figures')
    etl.robust_mkdir(dir_figs)
    path_head = os.path.join(dir_figs, str(visit_exam_id) + '_')
    return path_head


def chart_save_html(func, subj_series, params):

    # print the figure and save the results
    fig_path = chart_and_save(func, subj_series, params)
    # print and save that sweet video file of the eyes moving
    # video_path = replay_and_save(subj_series, params)
    # video = local_path_to_html_path(video_path)

    # create the html including the image and the video
    new_fig_html = get_exam_section_html(subj_series['exam'],
                                            figure=local_path_to_html_path(fig_path)
                                        )

    return new_fig_html



def visit_description_table(complete_df):
    table = complete_df.loc[0, ['subject_id',
                                'gender',
                                'date_of_birth',
                                'created_date_x',
                                'glasses',
                                'location',
                                'comments_visit',
                                'impairment',
                                'general comments']]
    table = table.append(pd.Series(complete_df.exam.str.cat(sep=', '), index=['exams']))
    string = table.to_frame().to_html(classes='center')
    return string


def get_exam_section_html(exam_name, **kargs):
    exam_header, footer = get_exam_header_footer_html(exam_name)

    # for including videos in the future
    # '''                    <video controls>
    #                     <source src="{{path_video}}" type="video/mp4">
    #                 </video> <br>'''

    fig_jinja = '''
                <div class="fig">
                    <img src="{{path_figure}}" alt="{{exam_name}}">
                </div>               
                '''
    fig_template = jinja2.Template(fig_jinja)
    figure_html = fig_template.render(
        exam_name=exam_name,
        path_figure=kargs['figure']
    )

    return exam_header + figure_html + footer


def get_exam_header_footer_html(exam_name):
    fig_jinja = '''
                    <div class="exam">
                    <!-- *** {{exam_name}} *** --->
                    <h1>{{exam_name}}</h1>
                '''
    fig_template = jinja2.Template(fig_jinja)
    header = fig_template.render(exam_name=exam_name)
    footer = '''
                </div><hr>


            '''
    return header, footer


def html_start():
    # HTML REPORT
    string = '''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <style type="text/css">
            .center {
                display: block;
                margin-left: auto;
                margin-right: auto;
                }
                
            h1 {
                text-align:center;
                }
                
            .fig img, video{
                width: 75%;
                max-width: 800px;
                margin: 0 auto;
                display: block;
                }
                
            table {
                    background-color:white;
                    border-color:black;
                    
                    }
            tr {
                border:none;
                }
            td, th {
                padding: 10px;
                border:none;
            }

            </style>
        </head>
        <body>
    '''
    return string


def html_end():
    html_string =  '''
                    </body>
                </html>
            '''
    return html_string



# def create_report(visit_id=None, params={}):
#
#     exam_to_vizfunc = {'prosaccade': chart.plot_prosaccade,
#                        'convergence': chart.plot_convergence,
#                        'pupillaryreflex': chart.plot_pupillaryreflex,
#                        'gaze': chart.plot_gaze,
#                        'smoothpursuit': chart.plot_smoothpursuit,
#                        'smoothpursuit2d': chart.plot_smoothpursuit2d,
#                        'trailmaking': chart.plot_trailmaking,
#                        'trailmaking2': chart.plot_trailmaking,
#                        'selfpacedsaccade': chart.plot_selfpacedsaccade,
#                        'categoryfluency': chart.plot_categoryfluency,
#                        'letterfluency': chart.plot_letterfluency,
#                        'stroop': chart.plot_stroop,
#                        'bostonnaming': chart.plot_bostonnaming,
#                        'digitspanforward': chart.plot_digitspanforward,
#                        'digitspanbackward': chart.plot_digitspanbackward,
#                        'tapping': chart.plot_tapping,
#                        'memoryencoding': chart.plot_memoryencoding,
#                        'memoryrecall': chart.plot_memoryrecall
#                        }
#
#
#     # set a default for the params to be AWS
#     if 'host' not in params:
#         params['host'] = 'aws'
#
#     processed_df = get_processed_df_for_visit(visit_id, params)
#
#     if len(processed_df) == 0:
#         logger.warning('No exams associated with visit_id: {}'.format(visit_id))
#
#     else:
#         # do charting and saving for each exam
#         if len(processed_df) > 0:
#             if visit_id is None:
#                 visit_id = processed_df.iloc[0]['visit_id']
#
#             full_html = html_start()
#             full_html += visit_description_table(processed_df)
#
#             for idx, subj_series in processed_df.iterrows():
#
#                 cur_exam = subj_series['exam'].lower()
#                 if cur_exam in exam_to_vizfunc:
#
#                     logger.info("PROCESSING: " + subj_series['exam'])
#
#                     func = exam_to_vizfunc[cur_exam]
#                     html_fig = chart_save_html(func, subj_series, params)
#                     full_html += html_fig
#
#             full_html += html_end()
#
#             host = udb.get_host()
#
#             dir_visit = get_dir_visit(visit_id)
#             path_html = os.path.join(dir_visit, visit_id + ".html")
#             etl.write_string(path_html, full_html)
#             if "aws" in params['host']:
#                 if host == 'production':
#                     # replace './reports' with 'production'
#                     path_bucket = path_html.replace(udf.PATH_REPORTS, udb.PATH_AWS_PRODUCTION)
#                 elif host == 'staging':
#                     path_bucket = path_html.replace(udf.PATH_REPORTS, udb.PATH_AWS_STAGING)
#
#                 udb.dump_file_s3(path_html, path_bucket)
#
#             if "local" not in params['host']:
#                 # clean up by removing the directory too
#                 # but only if at least one figure succeded
#                 etl.robust_rmdir(os.path.join(dir_visit, 'figures'))
#                 # remove the html and the directory from local
#                 os.remove(path_html)
#                 os.rmdir(os.path.join(dir_visit))
#
#         else:
#             logger.warning("No data to report for visit_id: " + visit_id)


def process_visit_exam(visit_exam_id, params):
    visit_exam_df = udb.get_visit_exams_by_id(visit_exam_id)
    visit_exam_processed_df = udb.load_visit_exams_processed(visit_exam_df, params=params)

    if len(visit_exam_processed_df) == 0:
        logger.warning('No exam associated with visit_exam_id: {}'.format(visit_exam_id))

    else:
        # do charting, saving and POST metrics for each exam
        visit_id = visit_exam_processed_df.iloc[0]['visit_id']

        subj_series = visit_exam_processed_df.iloc[0]
        cur_exam = subj_series['exam'].lower()

        control_subj_qty = 100 if 'control_subj_quantity' not in params else params['control_subj_quantity']
        year_range = 20 if 'year_range' not in params else params['year_range']
        skip_gender = False if 'skip_gender' not in params else params['skip_gender']
        params['overwrite'] = False if 'overwrite' not in params else params['overwrite']

        # get processed_df with controls, POST metrics, and chart
        if check_params_for_exam(subj_series, params):
            print("running: " + subj_series['exam'] + ", visit_exam: " + str(
                subj_series['visit_exam_id']) + ", visit_id: " + subj_series['visit_id'])

            # get the results from analysis for all rows
            complete_processed_df = ucon.load_processed_controls_from_pickle(subj_series, n=control_subj_qty,
                                                                             max_controls=False,
                                                                             year_range=year_range)
            if len(complete_processed_df) <= 10:
                logger.warning("Not enough control data to accurately calculate performance relative to population.")

            # POST metrics to database
            response = post_metrics(complete_processed_df, visit_id, cur_exam)

            # todo: check more error codes
            if '201' not in str(response):
                logger.warning('Metrics may not POST to database with error code {}.'.format(response))

            # do charting on the visit_exam
            if cur_exam in EXAM_TO_VIZ_FUNC:
                logger.info("PROCESSING: " + subj_series['exam'])

                func = EXAM_TO_VIZ_FUNC[cur_exam]
                fig = chart_and_save(func, complete_processed_df, params)

    return


def post_metrics(processed_df, visit_id, exam):
    # POST metrics to database
    metrics_to_post = extract_metrics_json(processed_df)

    host = udb.get_host()
    if host == 'staging':
        POST_URL = STAGING_POST_URL
    elif host == 'production':
        POST_URL = PRODUCTION_POST_URL
    else:
        POST_URL = STAGING_POST_URL

    url = POST_URL + "visits/{0}/exams/{1}/metrics".format(visit_id, exam)
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, data=json.dumps(metrics_to_post, cls=NumpyEncoder), headers=headers)

    logger.info("metrics POST for visit_id {0} returned response: {1}".format(visit_id, r))
    return r


# todo: need to actually dump into correct location
def extract_metrics_json(processed_df, params={}):
    # read in the document that has the metric names, bounds, units, etc.
    metric_ref = pd.read_csv('config/metrics_bounds.csv')
    metric_ref['exam'] = metric_ref['exam'].str.lower()

    # identify the visit_exam_id to look for in above document
    visit_exam_id = processed_df.iloc[-1]['visit_exam_id']
    exam = processed_df.iloc[-1]['exam'].lower()
    has_error = processed_df.iloc[-1]['processed']['has_error']

    # extract metrics from every row of processed_df
    # and make new metrics_df --> to make calculations easier
    key_cols = ['visit_exam_id', 'exam']
    metrics = []
    for idx, row in processed_df.iterrows():
        if 'metrics' in row['processed']:
            metric = row['processed']['metrics']
            for col in key_cols:
                metric[col] = row[col]
            metrics.append(metric)
    metrics_df = pd.DataFrame(metrics)

    # use all columns except visit_exam_id & exam and
    # calculate the general stats of the metrics
    cols_to_use = [w for w in metrics_df.columns if w not in key_cols]

    # identify each metric and pull data from metric_reference
    # and metrics_df to fill list of dicts
    all_metrics = []
    next_exam_params = None
    if has_error == False:
        for col in cols_to_use:
            # need to calculate/source numbers differently based on whether they're in the reference doc
            if (col not in metric_ref['name_internal'].values):
                if col != 'next_exam_params':
                    logger.warning("Metric: {} missing from spreadsheet. Please make sure metrics are consistent.".format(col))
                continue
            try:
                metric_dict = dict()
                metric_row = metric_ref.loc[metric_ref['exam'] == exam].loc[metric_ref['name_internal'] == col]
                metric_dict['name'] = metric_row['name_frontend'].iloc[0]
                if metric_row['unit'].iloc[0] == '#':
                    metric_dict['unit'] = None
                else:
                    metric_dict['unit'] = metric_row['unit'].iloc[0]

                low_perc = int(metric_row['red percentile'].iloc[0])
                high_perc = int(metric_row['green percentile'].iloc[0])
                better = metric_row['better'].iloc[0]

                # just in case the table hasn't been filled out/updated with the correct bounds
                if low_perc == high_perc:
                    low_perc = 25
                    high_perc = 75

                # get the version of the data science code from git version numbers
                metric_dict['version'] = ugit.get_tag()

                # calculate summary stats based on correct percentiles
                value = metrics_df.loc[metrics_df['visit_exam_id'] == visit_exam_id, col].iloc[0]
                if (type(value) == str) or (value is None):
                    metric_dict['value'] = value
                else:
                    # convert all numeric columns to float just to avoid any type errors later
                    metrics_df[col] = metrics_df[col].astype(np.float)
                    metrics_descrip = metrics_df[col].describe(percentiles=[low_perc / 100, high_perc / 100])
                    metric_dict['value'] = np.float(value)

                    min_value = np.float(metrics_descrip['min'])
                    max_value = np.float(metrics_descrip['max'])

                    # this determines whether we need to use the default values
                    # (in case not enough control data or something)
                    if min_value == max_value:
                        metric_dict['scale_min'] = np.float(metric_row['default_scale_min'])
                        metric_dict['scale_max'] = np.float(metric_row['default_scale_max'])
                    else:
                        metric_dict['scale_min'] = min_value
                        metric_dict['scale_max'] = max_value

                    # actually calculate result based on values in metrics_bounds.csv
                    metric_value = metric_dict['value']

                    if (better == 'higher' or better == 'lower') and (min_value != max_value):
                        low_value = metrics_descrip.loc[str(low_perc) + "%"]
                        high_value = metrics_descrip.loc[str(high_perc) + "%"]
                    #if there's not enough control data, just use the default values from the metrics_bounds.csv
                    elif (better == 'higher' or better == 'lower') and (min_value == max_value):
                        low_value = metric_row['default_low'].iloc[0]
                        high_value = metric_row['default_high'].iloc[0]

                # todo: actually calculate result legitimately
                result = 1

                metric_dict['result_level'] = result

                # replacing NaN w/None needs to be done for JSON serialization
                # pandas automatically converts None to NaN so it needs to be changed back
                metrics_df = metrics_df.replace({np.nan: None})
                if 'next_exam_params' in metrics_df.columns:
                    next_exam_params = metrics_df.loc[metrics_df['visit_exam_id'] == visit_exam_id, 'next_exam_params'].iloc[0]
                else:
                    next_exam_params = None

                # round to 4 decimal points so the API is happy
                metric_dict = round_metrics(metric_dict)

                all_metrics.append(metric_dict)
            except Exception as e:
                logger.error("Issue with metrics extraction for metric: {0}. Failed with exception: {1}".format(col, e))
    else:
        logger.error("Analysis for visit_exam_id={} has errors. Unable to produce metrics".format(visit_exam_id))

    metrics = {"metrics": all_metrics,
               "next_exam_params": next_exam_params}

    return metrics


def round_metrics(w):
    return {k: v if not etl.isdigit(str(v)) else np.around(v, decimals=4) for k, v in w.items()}


# https://pypi.org/project/numpyencoder/
# just took this from the source code of the above
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    """
    example query to run the report via command line:
    ipython report.py "81fbc179-9056-44d2-8fd8-c2f33e89c647" '{"videos": false, "host": "local"}'
    """
    if len(sys.argv) > 3:
        raise Exception('Please pass at most two parameters, a visit ID and parameters')
    else:
        print(sys.argv)
        visit_id = sys.argv[1]
        if len(sys.argv) > 2:
            params = json.loads(sys.argv[2])
            params['batch_begin_time'] = datetime.datetime.now(datetime.timezone.utc)

        else:
            # params = {"host": "local", "videos": False}
            params = {"videos": False, "host": "local", "skip_gender": True, "year_range": 100, "exams": "pupillaryreflex"}
        sys.exit(create_report(visit_id, params))
