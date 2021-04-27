# This file is meant to house tools for us to research metrics with



# make sure to set your host before calling these
# from utils_db import get_visit_exams, get_visits, pd_read_sql_conn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegressionCV, LogisticRegression


from matviz.matviz import etl
from matviz.matviz import viz
from matviz.matviz.histogram_utils import nhist
from matplotlib.pyplot  import xlabel, legend

import utils_db

from collections import defaultdict

# names of test sites

FNAME_PATIENT_DF = "../data/patient_df_03_21_2021.csv"


def get_metrics_data():
    sql_query = "select * from dbo.visit_exam_metrics \
            left join dbo.visit_exams on dbo.visit_exams.id=dbo.visit_exam_metrics.visit_exam_id \
            order by created_at desc"
    metrics_data = utils_db.pd_read_sql_conn(sql_query)
    metrics_data.drop(
        ['previous_result_level', 'previous_value', 'updated_at', 'visit_id', 'has_error', 'error_description'], axis=1,
        inplace=True)

    exam_mapper = etl.reverse_dict(utils_db.pd_get_id_map('exams'))
    metrics_data['exam'] = metrics_data['exam_id'].apply(lambda w: exam_mapper[w] if not pd.isnull(w) else 'NaN')
    location_mapper = etl.reverse_dict(utils_db.pd_get_id_map('locations'))
    metrics_data['location'] = metrics_data['location_id'].apply(
        lambda w: location_mapper[w] if not pd.isnull(w) else 'NaN')
    metrics_data = metrics_data.iloc[:, 1:]
    metrics_data.drop(['id'], axis=1, inplace=True)

    return metrics_data

def get_only_metrics():
#     created is missing because it varies across metric
    sql_query = "select visit_exam_id, name, value, result_level, version from dbo.visit_exam_metrics"
    metrics_data = utils_db.pd_read_sql_conn(sql_query)
    return metrics_data


def load_visit_exam_metrics(visit_exams_df, cur_exam):
    """
    return a copy of each metric in there, for a specific exam tho
    :param visit_exams_df:
    :param cur_exam:
    :return:
    """
    metrics_df = get_only_metrics()

    visit_exams_df_sub = visit_exams_df[visit_exams_df['exam'] == cur_exam]
    visit_exam_ids = set(visit_exams_df_sub['visit_exam_id'])

    # prepare for reordering the columns
    key_cols = list(metrics_df.columns)
    key_cols.remove('value')
    key_cols.remove('name')
    key_cols += ['name']

    # just pull out this exam's data
    metrics_df_sub = metrics_df[metrics_df['visit_exam_id'].apply(lambda w: w in visit_exam_ids)]
    # reorder the columns to put value last
    metrics_df_sub = metrics_df_sub[key_cols + ['value']]

    df_mult_idx = metrics_df_sub.set_index(key_cols, inplace=False)
    df_refact = df_mult_idx.unstack()['value'].reset_index()
    df_refact.columns.name = None

    return pd.merge(visit_exams_df_sub, df_refact, how='right', on='visit_exam_id')



def load_visits_metrics(visit_exams_df, cur_battery):
    """
    return a copy of each metric in there, for a specific exam tho
    :param visit_exams_df:
    :param cur_exam:
    :return:
    """

    # just select the visit_exams_df for the metrics we want
    cur_exams = set(battery_dict[cur_battery])
    visit_exams_df_sub = visit_exams_df[visit_exams_df['exam'].apply(lambda w: w in cur_exams)]
    visit_exam_ids = set(visit_exams_df_sub['visit_exam_id'])


    # get all the metrics
    metrics_df = get_only_metrics()
    # just pull out metrics from the exams from this battery data
    metrics_df_sub = metrics_df[metrics_df['visit_exam_id'].apply(lambda w: w in visit_exam_ids)]

    # expand the 'name' to include the exam
    visit_exam_id_to_exam = etl.dictify_cols(visit_exams_df_sub[['visit_exam_id', 'exam']])
    metrics_df_sub['name'] = metrics_df_sub.apply(
        lambda w: visit_exam_id_to_exam[w['visit_exam_id']] + "_" + w['name'].replace(" ", ""), axis=1)

    # add in the visit ID
    visit_exam_id_to_visit_id = etl.dictify_cols(visit_exams_df_sub[['visit_exam_id', 'visit_id']])
    metrics_df_sub['visit_id'] = metrics_df_sub['visit_exam_id'].apply(lambda w: visit_exam_id_to_visit_id[w])

    # reorder to prepare for unstacking
    key_cols = ['visit_id', 'name']
    metrics_df_sub = metrics_df_sub[key_cols + ['value']]

    # unstack by visit_id
    df_mult_idx = metrics_df_sub.set_index(key_cols, inplace=False)
    df_refact = df_mult_idx.unstack()['value'].reset_index()
    df_refact.columns.name = None

    # pd.merge(visit_exams_df_sub, df_refact, how='right', on='visit_id')
    return visit_exams_df_sub, df_refact



battery_dict = {
                'VOR1': ['Prosaccade', 'PupillaryReflex', 'SmoothPursuit'],
                'VOR2': ['Convergence', 'SelfPacedSaccades', 'SmoothPursuit2D'],
                'Neuro1': ['CategoryFluency', 'LetterFluency', 'TrailMaking', 'TrailMaking2'],
                'Neuro2': ['BostonNaming', 'DigitSpanBackwards', 'DigitSpanForwards', 'Stroop'],
                'Neuro3': ['CookieTheft', 'Memory', 'Tapping']
                }

# battery_dict['Neuro12'] = battery_dict['Neuro1'] + battery_dict['Neuro2']
# battery_dict['Neuro1_notrailsB'] = battery_dict['Neuro1'] + battery_dict['Neuro2']
# battery_dict['Neuro1_trailsB'] = battery_dict['Neuro1'] + battery_dict['Neuro2']
# battery_dict['...'] = battery_dict['Neuro1'] + battery_dict['Neuro2']


def get_common_symptoms(fname = FNAME_PATIENT_DF):
    df = pd.read_csv(fname)
    all_diagnoses = df['Diagnoses (List)'].values
    all_diagnoses = all_diagnoses[np.logical_not(pd.isnull(all_diagnoses))]
    all_str = ", ".join(all_diagnoses)
    lister = all_str.split(",")
    lister2 = [w[1:] if w[0] == ' ' else w for w in lister if len(w) > 0]
    lister3 = [w[:-1] if w[-1] == ' ' else w for w in lister2 if len(w) > 0]
    lister4 = [w.lower() for w in lister3]
    df = pd.DataFrame(etl.most_common(lister4),
                      columns=['condition', 'frequency'])  # .to_csv("../data/condition_frequency.csv")
    return df

def get_symptoms(fname = FNAME_PATIENT_DF):
    df = pd.read_csv(fname)
    all_diagnoses = df['Diagnoses (List)'].values
    all_diagnoses = all_diagnoses[np.logical_not(pd.isnull(all_diagnoses))]
    all_str = ", ".join(all_diagnoses)
    lister = all_str.split(",")
    lister2 = [w[1:] if w[0] == ' ' else w for w in lister if len(w) > 0]
    lister3 = [w[:-1] if w[-1] == ' ' else w for w in lister2 if len(w) > 0]
    lister4 = [w.lower() for w in lister3]
    df = pd.DataFrame(etl.most_common(lister4),
                      columns=['condition', 'frequency'])  # .to_csv("../data/condition_frequency.csv")
    return df

def parse_symptom_list(cur_str):
    lister = cur_str.split(",")
    lister2 = [w[1:] if w[0] == ' ' else w for w in lister if len(w) > 0]
    lister3 = [w[:-1] if w[-1] == ' ' else w for w in lister2 if len(w) > 0]
    lister4 = [w.lower() for w in lister3]
    if np.sum(['alz' in w for w in lister4]) > 0:
        lister4.append("alzheimer's")
    return lister4

def pca_er(df_refact):

    SKIP_THRESH = 0.5
    cols_to_skip = [k for k in df_refact.columns if  sum(pd.isnull(df_refact[k]))/len(df_refact) > SKIP_THRESH]
    cols_to_skip += ['TrailMaking_NumberCorrect', 'PupillaryReflex_PupilAsymmetry']

    cols_2_log = ['TrailMaking_RepeatCount',
                  'TrailMaking_ErrorCount',
                  'TrailMaking_TotalTime',
                  'CategoryFluency_NumberofIntrusions',
                  'LetterFluency_NumberofIntrusions',
                  'CategoryFluency_NumberofRepeats',
                  'LetterFluency_NumberofRepeats']

    cols_to_skip = [w for w in cols_to_skip if w in df_refact.columns]

    key_df = df_refact.drop(cols_to_skip, axis=1, inplace=False).dropna()

    X = []
    for ii, col in enumerate(key_df.columns[1:]):
        x = key_df[col].astype(float)
        if col in cols_2_log:
            x = np.log10(x + 1)
        X.append(x)

    X = np.array(X)

    cur_pca = PCA(n_components=X.shape[0])
    cur_pca.fit(X.T)
    Y = cur_pca.transform(X.T)

    coeff_df = pd.DataFrame(cur_pca.components_, columns=key_df.columns[1:])

    return coeff_df, Y, key_df

def prep_ml(df_refact):

    SKIP_THRESH = 0.5
    cols_to_skip = [k for k in df_refact.columns if  sum(pd.isnull(df_refact[k]))/len(df_refact) > SKIP_THRESH]
    cols_to_skip += ['TrailMaking_NumberCorrect', 'PupillaryReflex_PupilAsymmetry']

    cols_2_log = ['TrailMaking_RepeatCount',
                  'TrailMaking_ErrorCount',
                  'TrailMaking_TotalTime',
                  'CategoryFluency_NumberofIntrusions',
                  'LetterFluency_NumberofIntrusions',
                  'CategoryFluency_NumberofRepeats',
                  'LetterFluency_NumberofRepeats']

    cols_to_skip = [w for w in cols_to_skip if w in df_refact.columns]

    key_df = df_refact.drop(cols_to_skip, axis=1, inplace=False).dropna()

    X = []
    for ii, col in enumerate(key_df.columns[1:]):
        x = key_df[col].astype(float)
        if col in cols_2_log:
            x = np.log10(x + 1)
        X.append(x)

    X = np.array(X)

    return X, key_df, key_df.columns[1:].values


def parse_meta_df(fname = FNAME_PATIENT_DF):

    def parse_activity(w):
        if pd.isnull(w):
            return []
        else:
            return w.lower().split(",")


    def parse_diagnoses(w):
        lister = parse_activity(w)
        lister2 = [w[1:] if w[0] == ' ' else w for w in lister if len(w) > 0]
        lister3 = [w[:-1] if w[-1] == ' ' else w for w in lister2 if len(w) > 0]
        if np.sum(['alz' in w for w in lister3]) > 0:
            lister3.append("alzheimer's")
        return lister3


    def parse_general(w):
        if pd.isnull(w):
            return []
        else:
            w = w.replace('Other: DIABETIC', 'Diabetic')
            return [w]


    def parse_exercise(w):
        w = parse_diagnoses(w)
        w = [u.replace('sedentary', 'none') for u in w]
        w = [u.replace('some walking', 'walking') for u in w]
        w = [u.replace('walking', 'walk') for u in w]
        w = [u.replace('occasional walk', 'walk') for u in w]
        w = [u.replace('aerobic (walk)', 'walk') for u in w]
        w = [u.replace('walk and excersise', 'walk') for u in w]
        w = [u.replace('exercise bike', 'bike') for u in w]
        w = [u.replace('strech', 'stretch') for u in w]
        w = [u.replace('stretching', 'stretch') for u in w]
        return w


    def parse_eye(w):
        #     add 'eye' to the front of the color
        w = parse_general(w)
        return ['eye_' + u for u in w]


    def parse_education(w):
        edu_map = {
            'high school': 1,
            'college degree': 2,
            'college grad': 2,
            'education level': -1,
            "bachelor's": 2,
            "master's": 3,
            'phd': 4
        }

        if type(w) == str:
            w = edu_map[w.lower()]

        return w


    def parse_elopement(w):

        elope_map = {
            '0.0': 0,
            '0': 0,
            '3.0': 3,
            '1.0': 1,
            'low risk': 1.5,
            '2.0': 2,
            '10.0': 10,
            'no': 0,
            '1-2, low': 1.5,
            '3-30, high': 3,
            'high risk': 3,
            'low risk ': 1.5,
            'at risk': 2,
            'low risk ': 1.5,
            'no': 0,
            'yes': 3,
            'elopement risk': -1,
        }

        if type(w) == str:
            w = elope_map[w.lower()]

        return w


    def parse_fallrisk(w):
        fall_map = {
            '5.0': 5,
            '3.0': 3,
            '6.0': 6,
            'moderate': 5,
            '4.0': 4,
            '1.0': 1,
            '5-mod': 5,
            '7.0': 7,
            '2.0': 2,
            '9.0': 9,
            '7-mod': 7,
            '8.0': 8,
            '0.0': 0,
            '7': 7,
            '4': 4,
            '2-mod': 2,
            '8-mod': 8,
            '12-migh': 12,
            '10-migh': 10,
            '10.0': 10,
            '11.0': 11,
            '12.0': 12,
            'low': 1,
            'high': 12,
            'no': 0,
            'regular': 5,
            'yes': 7,
            'calculus of kidney': -1,
            'no concentrated sugar': -1,
            'no added salt': -1,
            'diabetic': -1,
            'fall risk': -1,

        }

        if type(w) == str:
            w = fall_map[w.lower()]

        return w


    def parse_nums(w):
        try:
            return float(w)
        except:
            return -1


    def parse_mmse(w):
        if pd.isnull(w):
            return w
        else:
            try:
                return float(str(w).replace("/30", "").replace(" out of 30", ""))
            except:
                return -1


    def parse_pt(w):
        pt_map = {
            'neither': 'therapy_None',
            'pt': 'therapy_PT',
            'both': 'therapy_PT_OT',
            '0.0': 'therapy_None',
            'ot': 'therapy_OT',
            'insomnia': 'ERROR',
            'pt/ot': 'therapy_PT_OT',
        }

        if type(w) == str:
            w = [pt_map[w.lower()]]
        else:
            w = []

        return w


    def parse_religion(w):
        religion_map = {
            'yes': 'religion_yes', 'no': 'religion_no'
        }

        if type(w) == str:
            if len(w) > 10:
                w = 'religion_error'
            else:
                w = [religion_map[w.lower()]]
        else:
            w = []

        return w


    def split_cat(w, sep="/"):
        w = [u.split(sep) for u in w]
        if len(w) > 0:
            w = np.concatenate(w)
        return w


    def parse_suppliment(w):
        w = parse_diagnoses(w)
        w = split_cat(w, "/")
        w = split_cat(w, "+")
        w = [u.replace('vitamin ', '') for u in w]
        w = [u.replace('vit ', '') for u in w]
        w = [u.replace('vitamins ', '') for u in w]
        w = [u.replace('eye vitamin', 'eye vitamins') for u in w]
        w = [u.replace('eye vitaminss', 'eye vitamins') for u in w]
        w = [u.replace(' d', 'd') for u in w]

        if 'vitamin' in w:
            w.remove('vitamin')

        return w

    key_symptoms = {'Activities (List)': parse_activity,
                    'Diagnoses (List)': parse_diagnoses,
                    'Diet': parse_general,
                    'Exercise (List)': parse_exercise,
                    'Eye Color': parse_eye,
                    'Medication Categories (List)': parse_diagnoses,
                    'PT/OT': parse_pt,
                    'Religion/Spirituality': parse_religion,
                    'Supplement Categories (List)': parse_suppliment
                    }

    key_ordinals = {
                    'Education Level': parse_education,
                    'Elopement Risk': parse_elopement,
                    'Fall Risk': parse_fallrisk,
                    'GDS': parse_nums,
                    'MMSE': parse_mmse,
                   }

    df = pd.read_csv(fname)

    df_sympt = pd.DataFrame(index=df.index)
    df_sympt['subject_id'] = df['id']
    for k, func in key_symptoms.items():
        df_sympt[str(k).replace(" (List)", "").replace("/", "")] = df[k].apply(func)

    df_ord = pd.DataFrame(index=df.index)
    df_ord['subject_id'] = df['id']
    for k, func in key_ordinals.items():
        df_ord[str(k).replace(" (List)", "").replace("/", "")] = df[k].apply(func)

    return df_sympt, df_ord



def get_visit_exams_per_user(one_per_user = True):
    """
    Get the most recent exam of each type per user

    :return:
    """
    visit_exams_df = utils_db.get_visit_exams(utils_db.get_visits(n=np.inf))
    # remove anyone who is a test subject
    test_subjects = utils_db.get_test_subjects()
    visit_exams_df = visit_exams_df[visit_exams_df['subject_id'].apply(lambda w: w not in test_subjects)]

    # pick just the most recent exam for each visit exam type for each subject
    # sort the exams by date
    visit_exams_df = visit_exams_df.sort_values('created_date_visit', axis=0)
    # jut pick the last exam that each user took
    if one_per_user:
        visit_exams_df = visit_exams_df.groupby(['subject_id', 'exam']).last().reset_index()

    return visit_exams_df


def get_key_Xy(cur_battery='Neuro1'):

    # the data you will be using
    # todo: make a mapping between this battery tag and specific lists of exams
    # that can span across different batteries
    # cur_battery = 'Neuro1'
    # cur_battery = 'Neuro1'
    # the key ordinal variable you want to build a model for
    key_ord = 'GDS'
    # key_ord = 'MMSE'



    visit_exams_df = get_visit_exams_per_user()
    # get age, subject, gender mapping
    visit_2_age = etl.dictify_cols(visit_exams_df[['visit_id', 'age']])
    visit_2_subject = etl.dictify_cols(visit_exams_df[['visit_id', 'subject_id']])
    visit_2_gender = etl.dictify_cols(visit_exams_df[['visit_id', 'gender']])
    visit_2_date = etl.dictify_cols(visit_exams_df[['visit_id', 'created_date_visit']])

    for k, v in visit_2_gender.items():
        if v == -1:
            visit_2_gender[k] = np.nan

    # load in the metadata
    df_sympt, df_ordinal = parse_meta_df()
    # metrics_df = get_only_metrics()

    visit_exams_df_sub, df_refact = load_visits_metrics(visit_exams_df, cur_battery)

    # Get the metrics into nice vectors
    X, key_df, key_cols = prep_ml(df_refact)
    X = X.T

    key_df['age'] = key_df['visit_id'].apply(lambda w: visit_2_age[w])
    key_df['subject'] = key_df['visit_id'].apply(lambda w: visit_2_subject[w])
    key_df['gender'] = key_df['visit_id'].apply(lambda w: visit_2_gender[w])
    key_df['created_date_visit'] = key_df['visit_id'].apply(lambda w: visit_2_date[w])

    # subject = key_df['subject'].values
    age = key_df['age'].values
    gender = key_df['gender'].values
    # date = key_df['created_date_visit'].values


    thresh_map = {'GDS': 2,
                  'MMSE': 24}

    threshold = thresh_map[key_ord]

    # Just select subjects with key data
    mapper = etl.dictify_cols(df_ordinal[['subject_id', key_ord]])
    key_y = key_df['subject'].apply(lambda w: mapper[w] if w in mapper else np.nan).values
    #     y = key_df.loc[:, col].values.astype(float)
    I = (~pd.isnull(key_y) * ~pd.isnull(age)).astype(bool)
    # scale the data based on those not used in the dataset
    # X = StandardScaler().fit_transform(X)

    scl = StandardScaler()
    # scl = MinMaxScaler()
    # scl = preprocessing.normalize()
    # scl.fit(X)
    scl.fit(X[~I])
    X = scl.transform(X)

    key_y = key_y[I] > threshold
    key_X = X[I, :]

    return key_X, key_y, key_df[I]

def check_logit_accuracy(key_X, key_y):
    # number of randomized train/test sets to test and aggregate results for
    num_train_test_sets = 200

    all_preds = []
    all_trues = []

    for random_state in range(num_train_test_sets):
        #     X_train, X_test, y_train, y_test = train_test_split(key_X, key_y, test_size=0.05, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(key_X, key_y, test_size=0.25, random_state=random_state)
        #
        # scl = StandardScaler()
        # scl.fit(X_train)
        # X_train = scl.transform(X_train)
        # X_test = scl.transform(X_test)

        if sum(y_train) >= 2:
        # skipping the test/train set ifthe train set doesn't have enough postive examples
            #         logit regression
            clf = LogisticRegression(C=1, max_iter=1000)
            clf.fit(X_train, y_train)

            key_pred = clf.predict_proba(X_test)
            key_pred = np.array([w[0] for w in key_pred])

            all_preds.append(key_pred)
            all_trues.append(y_test)


    viz.subplotter(1, 2, 0)
    auc = viz.plot_ROC(~np.concatenate(all_trues), np.concatenate(all_preds))
    
    viz.subplotter(1, 2, 1)
    y1 = np.concatenate(all_preds)[np.concatenate(all_trues)]
    y2 = np.concatenate(all_preds)[~np.concatenate(all_trues)]

    nhist({'y1': y1, 'y2': y2}, same_bins_flag=True, f=3)
    xlabel('predicted probability')
    viz.nicefy()
    legend(loc='best')


    return auc

  
def check_logit_accuracy_loo(key_X, key_y):
    # number of randomized train/test sets to test and aggregate results for

    all_preds = []
    all_trues = []

    loo = LeaveOneOut()
    # loo.get_n_splits(key_X)
    for train_index, test_index in loo.split(key_X):
        X_train, X_test = key_X[train_index], key_X[test_index]
        y_train, y_test = key_y[train_index], key_y[test_index]

        clf = LogisticRegression(C=1, max_iter=1000)
        clf.fit(X_train, y_train)

        key_pred = clf.predict_proba(X_test)
        key_pred = np.array([w[0] for w in key_pred])

        all_preds.append(key_pred)
        all_trues.append(y_test)

    viz.subplotter(1, 2, 0)
    auc = viz.plot_ROC(~np.concatenate(all_trues), np.concatenate(all_preds))
    viz.subplotter(1, 2, 1)
    y1 = np.concatenate(all_preds)[np.concatenate(all_trues)]
    y2 = np.concatenate(all_preds)[~np.concatenate(all_trues)]

    nhist({'y1': y1, 'y2': y2}, same_bins_flag=True, f=3)
    xlabel('predicted probability')
    viz.nicefy()
    legend(loc='best')


    return auc


def get_subj_to_battery(visit_exams_df, pass_partial=False):
    """
    Figure out which batteries these subjects have taken
    """
    subj_to_battery = defaultdict(list)
    for subj in set(visit_exams_df['subject_id']):
        df = visit_exams_df[visit_exams_df['subject_id'].apply(lambda w: w == subj)][['subject_id', 'exam']]

        for k, v in battery_dict.items():
            if np.any([w in df['exam'].values for w in v]):
                #  you at least got one, or you got them all
                if pass_partial or np.all([w in df['exam'].values for w in v]):
                    subj_to_battery[subj].append(k)

    return subj_to_battery


def deanonymize_subjects():
    """
    Get all the ids of the test subjects from both:
        * named "test" as a first name
        * anyone who has ever been tested at a test site

    :return: set of all subject ids who are in test
    """

    query = """
    SELECT id, first_name, last_name FROM dbo.{};
    """.format('subjects')
    df = utils_db.pd_read_sql_conn(query)
    df['full_name'] = df['first_name'] + ' ' + df['last_name']

    subj_to_name = etl.dictify_cols(df[['id', 'full_name']])

    cur_map = etl.reverse_dict(utils_db.pd_get_id_map('locations'))

    # test_location_names = set([cur_map[w] for w in TEST_LOCATION_NAMES])

    query = """
            SELECT id, subject_id, location_id 
            FROM dbo.visits
        """
    df = utils_db.pd_read_sql_conn(query)

    subj_to_location = defaultdict(str)
    for idx, row in df.iterrows():
        subj_to_location[row['subject_id']] = cur_map[row['location_id']]

    return subj_to_location, subj_to_name


def get_subjects_to_examinate(visit_exams_df, df_ordinal, key_ord, pass_partial=True):

    """

    :param visit_exams_df:
    :df_ordinal: the dataframe with subject metadata that is ordinal, like GDS
    :key_ord: str,  "GDS" as an example, or "MMSE"
    :pass_partial: boolean, if you want to count folks with at least an attempt at a battery
                    False if you want to require complete batteries
    :return:
            subjs_to_examinate: folks who have metadata, we should get them to take more exams
             subjs_to_hound: folks who have taken a lot of exams but we need their metadata
    """
    subj_to_battery = get_subj_to_battery(visit_exams_df, pass_partial=pass_partial)
    subj_to_location, subj_to_name = deanonymize_subjects()

    subjects_by_battery = pd.DataFrame(
        {k: v for k, v in sorted(subj_to_battery.items(), key=lambda item: len(item[1]), reverse=True)}.items(),
        columns=['subject_id', 'batteries'])
    subjects_by_battery['name'] = subjects_by_battery['subject_id'].apply(lambda w: subj_to_name[w])
    subjects_by_battery['location'] = subjects_by_battery['subject_id'].apply(lambda w: subj_to_location[w])

    #     get the subjects who have ordinal metadata
    subjs_with_ordinal_data = set(df_ordinal[['subject_id', key_ord]].dropna()['subject_id'].values)

    #     get the subjects that you want to take more exams
    subjs_to_examinate = subjects_by_battery[
        subjects_by_battery['subject_id'].apply(lambda w: w in subjs_with_ordinal_data)]

    #     get the subjects you want their metadata plzzz
    subjs_to_hound = subjects_by_battery[
        subjects_by_battery['subject_id'].apply(lambda w: w not in subjs_with_ordinal_data)]

    subjs_to_hound['num'] = subjs_to_hound['batteries'].apply(len)

    return subjs_to_examinate, subjs_to_hound

