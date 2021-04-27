"""Basic PDF report generation from a Pug template with pdf_reports.

A HTML page is generated from a template and rendered as a local PDF file.
"""

from pdf_reports import pug_to_html, write_report, preload_stylesheet

def create_report_summary(variables):

    variables = {
        'title': 'Brain Health Report',
        'logo': '/Users/suhaas/Desktop/ReactN/analytics/reactrack/REACT_logo.png',
        'exam_history_and_map_graph': '/Users/suhaas/Desktop/ReactN/analytics/test.png',
        'user_id': 'random_hash_basically',
        'name': 'Mary Smith',
        'age': '55',
        'location': 'LCB - Norwood',
        'date': '4/26/2020 1:11p',
        'VOR_phase_lag': 0.5,
        'VOR_bpm': 100,
        'VOR_nystagmus': 'No',
        'default_color': 'teal',
        'self_paced': None,
        'prosaccade_reaction_time_color': 'red',
        'prosaccade_acc_color': 'green',
        'prosaccade_catch_up_color': 'purple'
    }

    html = pug_to_html('./VOR_battery2_template.pug', **variables)
    write_report(html, 'VOR_2_Summary_Example.pdf')

variables = {}
create_report_summary(variables)