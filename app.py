from flask import Flask
from flask_restful import Resource, Api

import sys
import utils_db as udb
import chart as uch
import utils_dashboard as udash
from report import process_visit_exam
# generate_domains
app = Flask(__name__)
api = Api(app)

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from matviz.matviz import etl

from utils_logger import get_logger
logger = get_logger(__name__, None, streaming=True, to_file=False, debug=False)


@app.route("/process")
def process_visit_exams_app():
    visit_exam_mid_progress = visit_exam_in_progress()
    html_out = ""
    if visit_exam_mid_progress:
        html_out += "Currently processing visit_exam_id = " + str(visit_exam_mid_progress) + "<br>" + str(get_num_visit_exams_left()) + " visit exams left to go"
        html_out += "<br><br>Visit exams left to process: <br>" + str(get_visit_exams_left()).replace("\n", "<br>\n")
    else:
        html_out += process_visit_exams()

    quick_view, _ = udash.get_quick_view_progress(timeframe=['CURRENT_DATE'], drop_test=False)
    html_out += "<br><br>Quick Diagnostic <br>" + quick_view.to_html()

    return html_out

def process_visit_exams():

    next_visit_exam = get_next_visit_exam_to_process()

    if next_visit_exam is not None:
        visit_exam_id = next_visit_exam['id']
        visit_id = next_visit_exam['visit_id'] # TODO: DOES THIS NEED TO BE SET ANYWHERE ELSE?
        prev_visit_exam_id = visit_exam_id
    else:
        visit_exam_id = None

    while next_visit_exam is not None:
        visit_exam_id = next_visit_exam['id']
        params = dict() # TODO: CHECK WHERE PARAMS EVEN COMES FROM
        params['host'] = 'aws'

        logger.info("Processing: " + str(visit_exam_id))

        db_visit_exam_process_update('processing', 'start', visit_exam_id)
        # stop the previous visit from processing *after
        if prev_visit_exam_id != visit_exam_id:
            db_visit_exam_process_update('processing', 'stop', prev_visit_exam_id)

        # do the processing here for this visit
        try:
            process_visit_exam(visit_exam_id, params)
            # todo: generate domains on a visit level
            # generate_domains(visit_id, params)
        except Exception as e:
            logger.exception("Processing FAILED for " + str(visit_exam_id))

        # set the value to be true
        db_visit_exam_process_update('processed', 'start', visit_exam_id)

        prev_visit_exam_id = visit_exam_id
        next_visit_exam = get_next_visit_exam_to_process()

    # you are done, reset the processing flag (and if it isn't the first time you are running here
    if visit_exam_id is not None:
        db_visit_exam_process_update('processing', 'stop', prev_visit_exam_id)

    logger.info("finished processing all visit_exams")
    update_visits_table()
    # just a backup - should not be needed
    if visit_exam_in_progress():
        logger.warning("visit_exam should not have been mid process at this stage")
        reset_exam_processing()

    return "processing visit_exams complete"


def get_next_visit_exam_to_process():
    # Get the most recent visit - not processed, and not in processing
    # returns a data frame
    query = """
                SELECT id, visit_id, visit_exam_processed, created_date
                FROM dbo.visit_exams
                WHERE exam_version IS NOT NULL
                AND visit_exam_processed = false
                AND visit_exam_processing = false
                AND status IS NULL
               
                ORDER BY created_date ASC LIMIT 1;
            """

    # TODO: add in audio_transcribed=true to sql query
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)

    # validate the output
    if len(df) == 1:
        visit_exam_series = df.iloc[0]
        #visit_exam_series = visit_exam_series.insert(params=json.loads(visit_exam_series['reports_json']))
        # visit_exam_series = visit_exam_series.append(pd.Series({'params': json.loads(visit_exam_series['reports_json'])}))
        #visit_exam_series['params'] = json.loads(visit_exam_series['reports_json'])
    elif len(df) > 1:
        logger.error("number of visit exams is greater than one, check the 'get_next_visit_exam_to_process' sql code")
    else:
        logger.info("no visits left to process")
        visit_exam_series = None

    return visit_exam_series


def visit_exam_in_progress():
    query = """
                SELECT id FROM dbo.visit_exams
                WHERE visit_exam_processing = true
            """
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)
    conn.close()

    nprocessing = len(df)

    if nprocessing == 0:
        return False
    elif nprocessing in [1, 2]:
        return df.iloc[0]['id']
    else:
        raise Exception("{} visit_exams getting processed at once".format(str(nprocessing)))

def reset_exam_processing():
    """
    Set all of the processing values to false upon startup
    """
    query = """
            UPDATE dbo.visit_exams 
            SET visit_exam_processing = false
            WHERE visit_exam_processing = true;
            """

    udb.run_execute(query)

def db_visit_exam_process_update(processed_or_ing, start_stop, visit_exam_id):

    if start_stop == 'start':
        value = 'true'
    elif start_stop == 'stop':
        value = 'false'
    else:
        logger.error("unacceptable start_stop input")

    query = """
            UPDATE dbo.visit_exams 
            SET visit_exam_{0} = {1}
            WHERE id = '{2}' 
            """.format(processed_or_ing, value, visit_exam_id)

    udb.run_execute(query)

    return query


def get_visit_exams_left():
    query = """
                SELECT id FROM dbo.visit_exams
                WHERE visit_exam_processed = false
                AND visit_exam_processing = false
                AND exam_version IS NOT NULL
                AND status IS NULL

            """
    conn = udb.get_conn()
    df = pd.read_sql(query, conn)
    return df['id'].values

def get_num_visit_exams_left():
    query = """
                SELECT count(*)
                FROM dbo.visit_exams
                WHERE visit_exam_processed = false
                AND visit_exam_processing = false
                AND exam_version IS NOT NULL
                AND status IS NULL
            """

    conn = udb.get_conn()
    df = pd.read_sql(query, conn)
    return df['count'].values[0]

# TODO: should this be set to true when all visit_exams have processed or just at least one?
# thinking it should be reports_processed=true when at least one visit_exam is processed (this accounts for visits with skipped/terminated exams)
def update_visits_table():
    # update the visits table and set reports_processed=true where all associated
    # visit_exams have been processed
    query = """
                UPDATE dbo.visits
                SET reports_processed=true
                WHERE id IN
                (
                    SELECT DISTINCT(visit_id)
                    from dbo.visit_exams
                    WHERE visit_exam_processed=true
                )
               """
    udb.run_execute(query)


# def process_visits_app():
#     visit_mid_progress = visit_in_progress()
#     html_out = ""
#     if visit_mid_progress:
#         html_out += "Currently processing visit_id=" + visit_mid_progress + "<br>" + str(get_num_visits_left()) + " left to go"
#         html_out += "<br><br>Left to process: <br>" + str(get_visits_left()).replace("\n", "<br>\n")
#         logger.info(html_out)
#     else:
#         html_out += process_visits()
#
#     quick_view, _ = udash.get_quick_view_progress(timeframe=['CURRENT_DATE'], drop_test=False)
#     html_out += "<br><br>Quick Diagnostic <br>" + quick_view.to_html()
#     return html_out
#
# @app.route("/monitoring")
# def monitor_error_app():
#
#     failed_visit_exams, html = udash.get_failed_visit_exams(timeframe=['CURRENT_DATE'])
#     if len(failed_visit_exams.index) > 0:
#         exams = np.unique(failed_exams['exam'])
#         locations = np.unique(failed_exams['location'])
#         failure_by_exam = {exam: failed_exams.loc[failed_exams['exam'] == exam].shape[0] for exam in exams}
#         failure_by_location = {location: failed_exams.loc[failed_exams['location'] == location].shape[0] for location in
#                                locations}
#     else:
#         html += "No failed visit_exams"
#     return html
#
# def process_visits():
#
#     next_visit = get_next_visits_to_process()
#
#     if next_visit is not None:
#         visit_id = next_visit['id']
#         prev_visit_id = visit_id
#     else:
#         visit_id = None
#
#     while next_visit is not None:
#         visit_id = next_visit['id']
#         params = next_visit['params']
#         params['host'] = 'aws'
#
#         logger.info("Processing: " + visit_id)
#
#         db_visit_process_update('processing', 'start', visit_id)
#         # stop the previous visit from processing *after
#         if prev_visit_id != visit_id:
#             db_visit_process_update('processing', 'stop', prev_visit_id)
#
#         # do the processing here for this visit
#         try:
#             create_report(visit_id, params)
#         except Exception as e:
#             logger.exception("Report creation FAILED for " + visit_id)
#
#         # set the value to be true
#         db_visit_process_update('processed', 'start', visit_id)
#
#         prev_visit_id = visit_id
#         next_visit = get_next_visits_to_process()
#
#     # you are done, reset the processing flag (and if it isn't the first time you are running here
#     if visit_id is not None:
#         db_visit_process_update('processing', 'stop', prev_visit_id)
#
#     logger.info("finished processing all visits")
#
#     # just a backup - should not be needed
#     if visit_in_progress():
#         logger.warning("visit should not have been mid process at this stage")
#         reset_processing()
#
#     return "processing visits complete"
#
#
# def visit_in_progress():
#     query = """
#                 SELECT id FROM dbo.visits
#                 WHERE reports_processing = true
#             """
#     conn = udb.get_conn()
#     df = pd.read_sql(query, conn)
#     conn.close()
#
#     nprocessing = len(df)
#
#     if nprocessing == 0:
#         return False
#     elif nprocessing in [1, 2]:
#         return df.iloc[0]['id']
#     else:
#         raise Exception("{} visits getting processed at once".format(str(nprocessing)))
#
#
# def reset_processing():
#     """
#     Set all of the processing values to false upon startup
#     """
#     query = """
#             UPDATE dbo.visits SET reports_processing = false
#             WHERE reports_processing = true;
#             """
#
#     udb.run_execute(query)
#
# def reset_batch_processing(params_json):
#     """
#     Set all of the processing values to false upon startup
#
#     for example:
#     {"videos": false, "host": "local", "exams":["Prosaccade", "convergence"]}
#     """
#
#     cur_params = json.loads(params_json)
#     cur_params['batch_begin_time'] = str(datetime.datetime.now(datetime.timezone.utc))
#     param_str = json.dumps(cur_params)
#
#     # update the parameter string to be what is specified
#     query = """
#                 UPDATE dbo.visits SET reports_json = (%s);
#             """
#     udb.run_execute(query, param_str)
#
#     # update all the exams to be specified as not having been processed so they get recomputed
#     query = """
#                 UPDATE dbo.visits SET reports_processed = false
#                 WHERE reports_processed = true;
#             """
#     udb.run_execute(query)
#
# def get_next_visits_to_process():
#     # Get the most recent visit - not processed, and not in processing
#     # returns a data frame
#     query = """
#                 SELECT id, reports_processed, created_date, reports_json
#                 FROM dbo.visits
#                 WHERE reports_processed = false
#                 AND reports_processing = false
#                 AND id NOT IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NULL
#                 )
#                 AND id IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NOT NULL
#                 )
#                 ORDER BY created_date ASC LIMIT 1;
#             """
#
#     conn = udb.get_conn()
#     df = pd.read_sql(query, conn)
#
#     # validate the output
#     if len(df) == 1:
#         visit_series = df.iloc[0]
#         #visit_series = visit_series.insert(params=json.loads(visit_series['reports_json']))
#         visit_series = visit_series.append(pd.Series({'params': json.loads(visit_series['reports_json'])}))
#         #visit_series['params'] = json.loads(visit_series['reports_json'])
#     elif len(df) > 1:
#         logger.error("number of visits is greater than one, check the 'get_next_visits_to_process' sql code")
#     else:
#         logger.info("no visits left to process")
#         visit_series = None
#
#     return visit_series
#
# def get_visits_to_process():
#     query = """
#                 SELECT id, reports_processed FROM dbo.visits
#                 WHERE reports_processed = false
#             """
#     conn = udb.get_conn()
#     df = pd.read_sql(query, conn)
#     return df['id'].values
#
#
# def get_num_visits_left():
#     query = """
#                 SELECT count(*)
#                 FROM dbo.visits
#                 WHERE reports_processed = false
#                 AND reports_processing = false
#                 AND id NOT IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NULL
#                 )
#                 AND id IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NOT NULL
#                 );
#             """
#
#     conn = udb.get_conn()
#     df = pd.read_sql(query, conn)
#     return df['count'].values[0]
#
#
# def get_visits_left():
#     query = """
#                 SELECT id FROM dbo.visits
#                 WHERE reports_processed = false
#                 AND reports_processing = false
#                 AND id NOT IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NULL
#                 )
#                 AND id IN
#                 (
#                     SELECT DISTINCT(visit_id)
#                     from dbo.visit_exams
#                     WHERE exam_version IS NOT NULL
#                 );
#             """
#     conn = udb.get_conn()
#     df = pd.read_sql(query, conn)
#     return df['id'].values
#
# def db_visit_process_update(processed_or_ing, start_stop, visit_id):
#
#     if start_stop == 'start':
#         value = 'true'
#     elif start_stop == 'stop':
#         value = 'false'
#     else:
#         logger.error("unacceptable start_stop input")
#
#     query = """
#             UPDATE dbo.visits SET reports_{0} = {1}
#             WHERE id = '{2}'
#             """.format(processed_or_ing, value, visit_id)
#
#     udb.run_execute(query)
#
#     return query

if __name__ == '__main__':
    # in case there were any visits stuck in processing if something crashed
    # reset them here when turning on the API
    reset_exam_processing()

    flask_config = udb.load_json("flask_config.json")
    app.run(debug=False, use_reloader=False, host=flask_config['host'], port=flask_config['port'])


