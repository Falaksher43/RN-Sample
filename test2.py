import utils_db as udb
import numpy as np
import datetime
import report
import os

def test_get_visits():
    udb.set_host("staging")

    # get N visits
    visit_df = udb.get_visits(n=13)
    assert len(visit_df) == 13

    # get specific visits:
    test_visit_ids = ['1b3cf7c8-e45c-41f3-87c3-c416c78f6305',
                      'be6b4716-db1b-42bc-b614-7982f4614deb',
                      '90a88ed5-0bf5-4ffd-b46c-91720d868cef',
                      '98315020-68a4-46e6-b07f-599acf0c507d',
                      '8e0444d4-9b86-47b4-a02c-d0df06005441',
                      '837edd81-4b82-4111-920b-46e2e2726aeb',
                      'fa65a8a0-425f-4cc4-9c75-a873627248c1',
                      'bded41a8-d478-4f96-9706-30a833e431b1',
                      '52929e2f-fa57-4e5a-a8aa-c2b890f4c200',
                      '971a1e47-dc29-48dd-b6fd-6ee3172fcb5c',
                      '20f61c59-0312-4cd5-b4c3-df34b5cfe204',
                      'cfd1f60b-5b3d-442b-a115-33b5023389c8',
                      '417a7d6e-75c7-4211-be0c-4d07a0f30258']

    test_subject_ids = ['66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        'b1d62c65-c6bb-4839-ac57-59ab4ea73fc7',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '66999214-048e-4a41-a51b-4e8197c2e668',
                        '675bcea5-2bbd-4cdd-b2b3-abae53b33a80',
                        '66999214-048e-4a41-a51b-4e8197c2e668']

    visit_df = udb.get_visits(visit_id=test_visit_ids)
    assert list(visit_df.visit_id.values) == test_visit_ids
    assert list(visit_df.subject_id.values) == test_subject_ids


def test_get_visit_exams():
#     test the various elements needed for app.py
    print("test loading visits")
    visit_id = "4347d97f-cd56-4865-a505-3506246e9ed7"

    expected_exams = ['TrailMaking2', 'CategoryFluency', 'LetterFluency', 'TrailMaking', 'Prosaccade']

    visit_exams_df = udb.get_visit_exams(udb.get_visits(visit_id=visit_id))
    # check that all 7 visit_exams get returned
    assert len(visit_exams_df) == 5
    # check that they are the correct ones
    assert all([w in expected_exams for w in visit_exams_df['exam']])


def test_load_processed():
    print("testing load processed")

    test_begin_time = datetime.datetime.now(datetime.timezone.utc)

    visit_id = "6feeef57-4047-4c2d-b5f1-7e02f60a0188"


    # todo: add each exam as it is finished
    visit_exams_df = udb.get_visit_exams(udb.get_visits(visit_id=visit_id))

    # filter out any we know haven't passed the test yet
    # todo: remove this filter because everything should pass the test
    exams_to_test = ['TrailMaking', 'TrailMaking2', 'CategoryFluency', 'LetterFluency']#, 'Prosaccade']
    visit_exams_df = visit_exams_df[visit_exams_df['exam'].apply(lambda w: w in exams_to_test)]

    # note: host is AWS so that this doesn't leave a mess
    params = {'videos': False,
              'host': 'aws',
              'control_subj_quantity': 2,
              'exams': exams_to_test,
              'overwrite': True}

    print("Test recomputing results")
    processed_df = udb.load_visit_exams_processed(visit_exams_df, params)
    # Check that all the status passed
    for idx, subj_series in processed_df.iterrows():
        # check that each exam
        assert 'has_error' in subj_series, "missing error status field"
        assert subj_series['has_error'] == False, "error occured in: " + subj_series['exam']
        assert 'has_error' in subj_series['processed'], "missing status element in exam: " + subj_series['exam']
        # check that the status is true
        assert subj_series['processed']['has_error'] == False, "processing has error in exam: " + subj_series['exam'] +\
                                                                        ", has_error: " + subj_series['processed']['has_error']

    # test that the processed files have all been modified by that process
    for idx, subj_series in processed_df.iterrows():
        response = udb.load_s3_object(udb.get_processed_path(subj_series))
        assert response['LastModified'] > test_begin_time, "overwrite: True, processed file was not modified when it should have been for exam: " + subj_series['exam']


    # tests below measure a few cases of the overwrite_now function
    # check that the files don't get reprocessed again if param is set to false
    test_begin_time = datetime.datetime.now(datetime.timezone.utc)
    params['overwrite'] = False
    processed_df2 = udb.load_visit_exams_processed(visit_exams_df, params)
    for idx, subj_series in processed_df2.iterrows():
        response = udb.load_s3_object(udb.get_processed_path(subj_series))
        assert response['LastModified'] < test_begin_time, "overwrite:'False' processed file was modified when it shouldn't have been for exam: " + subj_series['exam']

    # test batch overwrite: the files don't get reprocessed again for a long-running batch
    params['overwrite'] = 'batch'
    # pretend the batch started 2 hours ago
    test_begin_time = datetime.datetime.now(datetime.timezone.utc)
    params['batch_begin_time'] = test_begin_time - datetime.timedelta(hours=2)
    processed_df2 = udb.load_visit_exams_processed(visit_exams_df, params)
    for idx, subj_series in processed_df2.iterrows():
        response = udb.load_s3_object(udb.get_processed_path(subj_series))
        assert response['LastModified'] < test_begin_time, "overwrite:'batch' processed file was modified when it shouldn't have been for exam: " + subj_series['exam']

    # Test batch overwrite: the files *do* get reprocessed if older than when the batch started
    params['batch_begin_time'] = datetime.datetime.now(datetime.timezone.utc)
    processed_df2 = udb.load_visit_exams_processed(visit_exams_df, params)
    for idx, subj_series in processed_df2.iterrows():
        response = udb.load_s3_object(udb.get_processed_path(subj_series))
        assert response['LastModified'] > params['batch_begin_time'], "overwrite:'batch' processed file was not modified when it should have been for exam: " + subj_series['exam']

#     todo: test the 'number of days' overwrite parameter


def test_processing_functions():
    """
    test results of the processing functions

    :return:
    """
    visit_id = "6feeef57-4047-4c2d-b5f1-7e02f60a0188"

    # todo: add each exam as it is finished
    exams_to_test = ['TrailMaking2', 'CategoryFluency', 'LetterFluency', 'TrailMaking']#, 'Prosaccade']

    # note: assumption is that the processing was already re-done in the earlier test function.
    params = {'videos': False,
              'host': 'aws',
              'control_subj_quantity': 2,
              'exams': exams_to_test,
              'overwrite': False}

    visit_exams_df = udb.get_visit_exams(udb.get_visits(visit_id=visit_id))
    processed_df = udb.load_visit_exams_processed(visit_exams_df, params)

    # filter out any we know haven't passed the test yet
    # todo: remove this filter because everything should pass the test
    processed_df = processed_df[processed_df['exam'].apply(lambda w: w in exams_to_test)]

    for idx, subj_series in processed_df.iterrows():

        if subj_series['exam'] == 'TrailMaking':
            assert subj_series['processed']['metrics']['error_count'] == 13, "trailmaking produced wrong error count"
            assert subj_series['processed']['metrics']['repeat_count'] == 14, "trailmaking produced wrong repeat count"
            assert subj_series['processed']['active time'] == 22.0552, "trailmaking produced wrong active time"

        elif subj_series['exam'] == 'TrailMaking2':
            assert subj_series['processed']['metrics']['error_count'] == 16, "trailmaking2 produced wrong error count"
            assert subj_series['processed']['metrics']['repeat_count'] == 11, "trailmaking2 produced wrong repeat count"
            assert subj_series['processed']['active time'] == 19.5775, "trailmaking2 produced wrong active time"

        elif subj_series['exam'] == 'CategoryFluency':
            assert subj_series['processed']['data']['responses'].iloc[0]['transcript'] == 'cucumbers', "transcript first word incorrect"
            assert subj_series['processed']['data']['responses'].iloc[1]['transcript'] == 'carrots', "transcript second word incorrect"
            assert subj_series['processed']['data']['responses'].iloc[2]['transcript'] == 'celery', "transcript second word incorrect"
            assert subj_series['processed']['metrics']['num_correct'] == 6, "number correct wrong"

        elif subj_series['exam'] == 'LetterFluency':
            assert subj_series['processed']['data']['responses'].iloc[0]['transcript'] == 'apples'
            assert subj_series['processed']['data']['responses'].iloc[1]['transcript'] == 'asparagus'
            assert subj_series['processed']['data']['responses'].iloc[2]['transcript'] == 'australia'
            assert subj_series['processed']['metrics']['num_correct'] == 4



#
# def test_database_consistency():
#
#     # todo: fix this function when we have this data coming in.
#     # pick a user, or a few users
# #     hard-code the eye color of that user here
# #     test that each row returned has the correct eye color
# #     we can also include another field in this test, such as birthdate
#
#
#     test_gender_subject_ids = {'98986e3c-a346-4703-b6b5-d38fe35cb0f1': 1,
#                             'ede63b45-3586-4bcb-b389-d06609068d34': 1,
#                             '4929308a-dd03-47aa-b40a-602a564d62a8': 1,
#                             '6abe2f09-ffae-4aea-85d0-32280fab8161': 1}
#
#     test_dob_subject_ids = {'98986e3c-a346-4703-b6b5-d38fe35cb0f1': '1942-10-02 00:00:00',
#                          'ede63b45-3586-4bcb-b389-d06609068d34': '1924-04-08 00:00:00',
#                          '4929308a-dd03-47aa-b40a-602a564d62a8': '1996-08-27 00:00:00',
#                          '6abe2f09-ffae-4aea-85d0-32280fab8161': '1918-11-07 00:00:00'}
#
#     test_eye_subject_ids = {'4838c57d-12c6-4f04-ba17-dbe2c4f6b5e7': 'Brown',
#                             '0a62f288-dc11-4165-9cf1-6985953f5ec9': 'Blue',
#                             '4d29aa47-3129-49bb-a005-44524aca1056': 'Hazel',
#                             'dc56a992-6661-4b46-97f5-800c27232dc0': 'NaN'}
#
#     for key in test_dob_subject_ids:
#         user_df = udb.get_visit_exams(udb.get_visits(subject_id=key))
#         assert len(np.unique(str(user_df['date_of_birth']))) == 1, "DOB not consistent across visits for User " + key
#         assert str(user_df.iloc[0]['date_of_birth']) == test_dob_subject_ids[key], "DOB does not match test for User " + key
#
#
#     for key in test_gender_subject_ids:
#         user_df = udb.get_visit_exams(udb.get_visits(subject_id=key))
#         assert len(np.unique(str(user_df['gender']))) == 1, "Gender not consistent across visits for User " + key
#         assert user_df.iloc[0]['gender'] == test_gender_subject_ids[key], "Gender does not match test for User " + key
#
#     # for key in test_eye_subject_ids:
#     #     user_df = udb.get_visit_exams(udb.get_visits(subject_id=key))
#     #     assert len(np.unique(str(user_df['q_16']))) == 1, "Eye Color not consistent across visits for User " + key
#     #     assert user_df.iloc[0]['q_16'] == test_eye_subject_ids[key], "Eye Color does not match test for User " + key

#
# # def test_concatenate_controls():
# # todo: figure out how to test concatenate controls - when the controls could change
# #       maybe just set n=n.inf and then check for the existence of known controls depending on the parameters
#
#
#