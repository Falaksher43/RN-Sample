import os
import jinja2
import chart
import utils_df
from json2html import json2html

from matviz.matviz import etl


def load_jinja_template(template_path):
    with open(template_path) as file_:
        template = jinja2.Template(file_.read())
    return template


def get_summary_html(results):
    summary_html = """
                    <h2>Folder:</h2>
                    
                    """ + results['folder']

    summary_results = {k: v['summary'] for k, v in results.items() if k !='folder' and 'summary' in v}


    print(summary_results)
    #
    # to_skip = ['main_result', 'summary', 'reaction_times', 'median reaction time', 'duration', 'eye_timeseries',
    #            'stim_starts','stim_ends', 'eye_starts', 'eye_ends', 'eye_fixs', 'max_speeds']
    # for cur_exam in hlp.EXAM_LIST:
    #     if cur_exam in results:
    #         cur_result = results[cur_exam]
    #         # leave out the reaction time since it is too long
    #         # if cur_exam in ['prosaccade']:
    #         #     cur_result = {k: v for k, v in cur_result if k not in to_skip}
    #
    #         summary_html += "\n<h2>" + cur_exam + "</h2>\n"
    #
    #
    #         # for  k in cur_result.keys():
    #         #     if k not in to_skip:
    #         #         print(k)
    #         #         json2html.convert({k: cur_result[k]}) + "\n\n"
    #
    #
    #         summary_html += json2html.convert({k: v for k, v in cur_result.items() if k not in to_skip}) + "\n\n"

    summary_html += """
                    <h2>Sampling summary</h2>

                    """ + json2html.convert(summary_results)


    return summary_html


def get_fig_txt(cur_exam, cur_viz):
    fig_jinja = """
                    <div class="fig">
                    <img class="{{img_class}}" src="figs/{{path}}" alt="{{label}}">
                    <br/>
                    {{caption}}
                    </div>
                    <p><div></div>

                """

    fig_template = jinja2.Template(fig_jinja)

    viz_label = cur_viz.__name__
    viz_path = cur_exam + "_" + viz_label + ".png"
    viz_caption = viz_label.replace("_", " ")

    img_class = 'small' if (viz_label in utils_df.SMALL_FIG_LIST) else 'big'


    txt = fig_template.render(
        label=viz_label,
        caption=viz_caption,
        path=viz_path,
        img_class=img_class
    )

    return txt



def generate_html(cur_folder):

    # load in the results
    # use Jinja templating to put in the username etc.

    # load that answers json
    answers_path = os.path.join(cur_folder, "answers.json")
    if os.path.exists(answers_path):
        answers = etl.load_json(answers_path)
    else:
        answers = {
                    'Email?': '',
                    'Gender? (M/F)': '',
                    'Birthday? (M/D/YYYY)': '',
                    'Eye color?': '',
                    'Comments:': '',
                    'id': cur_folder.split("/")[-1],
                    'timestamp': '?',
                    'utc': '?'
                    }
    # load the results json
    results = etl.load_json(os.path.join(cur_folder, "results.json"))

    id = answers['id']
    trial_vizs = chart.get_exam_vizs_debug()
    all_fig_text = ""
    for cur_exam in utils_df.EXAM_LIST:
        all_fig_text += "<h2>" + cur_exam + "</h2>\n"
        viz_list = trial_vizs[cur_exam]
        print("Plotting figures for: " + cur_exam)
        for cur_viz in viz_list:
            all_fig_text += get_fig_txt(cur_exam, cur_viz) + "\n"

    # answers['summary'] = get_summary_html(results)
    answers['figures'] = all_fig_text

    template = load_jinja_template(utils_df.TEMPLATE_PATH)
    txt_str = template.render(**answers)

    template_path_dest = os.path.join(cur_folder, id + ".html")
    with open(template_path_dest, 'w') as f:
        f.write(txt_str)

    print("html file generated for: " + id)

    return template_path_dest


