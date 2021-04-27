import numpy as np
from matviz.matviz import etl
import utils_db
import chart


visit_exam_ids = [5776, 5987, 5779, 5985, 5986, 5724, 5723, 5988, 5857, 5778, 5688, 5990, 5991, 5498, 5729, 5989, 5488]


def get_test_visit_exams_df():

    visit_exams_df = utils_db.get_visit_exams_by_id(visit_exam_ids)

    return visit_exams_df


def get_test_with_controls_visit_exams_df():
    """
    get a visit exams dif that has all the key visit_exam_ids; as well as
    a specific set of 2-3 additional control visit_exams per exam
    :return:
    """
    pass
    # visit_exam_ids =
    #
    # visit_exam_df = utils_db.get_visit_exams_by_id(visit_exam_ids)

    # return visit_exam_df

def test_analysis():
    """

    The prep for this can be found in test_analysis.py

    :return:
    """
    expected_output = {
                        'PupillaryReflex': "{'pupil_max_dilation': 2.0675, 'pupil_max_constric': 0.04, 'pupil_range': 2.0583, 'pupil_dilation_velocity': -0.39, 'aniscoria': 'No', 'next_exam_params': None}",
                        'Convergence': "{'stimulus_phase_lag': 2.3588, 'stimulus_correlation': 0.2872, 'LR_phase_lag': 0.4175, 'LR_correlation': 0.5057, 'next_exam_params': None}",
                        'Prosaccade': "{'number_of_saccades': 43, 'saccades_per_movement': 1.23, 'median_reaction_time': 0.2, 'duration': 41.42, 'abnormal_path_proportion': 0.02, 'next_exam_params': None}",
                        'SmoothPursuit2D': "{'error_magnitude': 1.8545, 'error_angular': 18.8197, 'error_radial': 1.7848, 'next_exam_params': None}",
                        'SelfPacedSaccade': "{'sacc_per_sec': 6.18, 'median_vel': 23.53, 'vel_acc': 23.53, 'accuracy': 1, 'next_exam_params': None}",
                        'SmoothPursuit': "{'median_lag_left': 0.0708, 'median_lag_right': 0.085, 'next_exam_params': None}",
                        'CategoryFluency': "{'num_correct': 3, 'num_repeats': 0, 'num_intrusions': 1, 'next_exam_params': 'vegetables'}",
                        'Stroop': "{'speed_median': 0.6784, 'accuracy': 90.0, 'num_correct': 27, 'next_exam_params': None}",
                        'TrailMaking': "{'total_time': 12.1003, 'error_count': 12, 'repeat_count': 10, 'num_correct': 25, 'next_exam_params': None}",
                        'TrailMaking2': "{'total_time': 7.7975, 'error_count': 2, 'repeat_count': 12, 'num_correct': 25, 'next_exam_params': None}",
                        'LetterFluency': "{'num_correct': 1, 'num_repeats': 0, 'num_intrusions': 22, 'next_exam_params': 'c'}",
                        'BostonNaming': "{'speed_median': 0.7647, 'accuracy': 93.3333, 'num_correct': 14, 'next_exam_params': None}",
                        'DigitSpanForward': "{'max_level_perfect': 0, 'max_level': 5, 'next_exam_params': 4}",
                        'MemoryEncoding': "{'max_words_correct': 10, 'num_intrusions': 0, 'next_exam_params': 15}",
                        'Tapping': "{'right_section_right_presses': 88, 'right_section_left_presses': 1, 'left_section_right_presses': 0, 'left_section_left_presses': 86, 'alternate_section_right_presses': 74, 'alternate_section_left_presses': 66, 'ordering_errors': 8, 'next_exam_params': None}",
                        'DigitSpanBackward': "{'max_level_perfect': 0, 'max_level': 0, 'next_exam_params': 3}",
                        'MemoryRecall': "{'recall_num_correct': 0, 'recall_num_intrusions': 0, 'recognize_num_correct': 0, 'next_exam_params': None}"
                        }


    params = {"videos": False, "host": "local",
              "control_subj_quantity": 0,
              "overwrite": False}

    visit_exams_df = get_test_visit_exams_df()
    processed_df = utils_db.load_visit_exams_processed(visit_exams_df, params)
    output = {row['exam']: str(round_metrics(row['processed'])) for idx, row in processed_df.iterrows()}

    print("Checking database metrics haven't changed")
    check_output(output, expected_output)

    params = {"videos": False, "host": "local",
              "control_subj_quantity": 0,
              "overwrite": True}

    processed_df = utils_db.load_visit_exams_processed(visit_exams_df, params)
    output = {row['exam']: str(round_metrics(row['processed'])) for idx, row in processed_df.iterrows()}
    print("Checking reprocessing results in the same output")
    check_output(output, expected_output)


def check_output(output, expected_output):
    for k, v in expected_output.items():
        assert k in output, "Processed results results missing exam: " + k
        assert v == output[k], "Processed metrics don't match for exam: " + k

    return True


# todo: remove this once every processing output has round_metrics
def round_metrics(w):
    return {k: v if not etl.isdigit(str(v)) else np.around(v, decimals=4) for k, v in
                           w['metrics'].items()}


def test_charts():
    print("Imma testing chartssS!!!")

    visit_exam_df = get_test_visit_exams_df()

    params = {"videos": False, "host": "local",
              "control_subj_quantity": 0,
              "overwrite": False}

    processed_df = utils_db.load_visit_exams_processed(visit_exam_df, params)

    plot_funcs = processed_df['exam'].apply(lambda w: eval("chart.plot_" + w.lower())).values
    for ii in range(len(processed_df)):
        func = plot_funcs[ii]
        exam_id = processed_df['visit_exam_id'].iloc[ii]
        func(processed_df[ii:ii + 1], exam_id)


def todo_test_charts_with_controls():


    visit_exam_df = get_test_with_controls_visit_exams_df()

    params = {"videos": False, "host": "local",
              "control_subj_quantity": 0,
              "overwrite": False}

    processed_df = utils_db.load_visit_exams_processed(visit_exam_df, params)

    exam_to_chart = lambda w: eval("chart.plot_" + w.lower())

    # todo: check that this works that the groups are formatted correctly
    # select just one example of each exam (with pandas groupby?)
    groups = processed_df.groupby("exam")
    for exam, group in groups:
        func = exam_to_chart(group.iloc[0]['exam'])
        exam_id = processed_df['visit_exam_id'].iloc[0]
        func(group, exam_id)

# todo: figure out a way to test that the output is also the same

