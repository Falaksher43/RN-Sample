import streamlit as st
import pandas as pd
import numpy as np
import datetime
import utils_db as udb
import utils_dashboard as udash
import matplotlib.pyplot as plt
import copy
import seaborn as sns

st.title('REACT Data Dashboard')

# -------------------------------------------------------------------------------------------------------
# set host to see data from either production or from staging
host = st.sidebar.selectbox(
					'Database', 
					('Staging', 'Production'))

udb.set_host(host.lower())

# -------------------------------------------------------------------------------------------------------
# get date range and number of days that that represents (needed for utils_dashboard.py function)
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

end_date += datetime.timedelta(days=1)

str_start_date = datetime.date.strftime(start_date, '%m-%d-%Y')
str_end_date = datetime.date.strftime(end_date, '%m-%d-%Y')

timeframe = [str_start_date, str_end_date]

success_or_failed = st.sidebar.selectbox(
					'Status',
					('Successful', 'Failed'))

# -------------------------------------------------------------------------------------------------------
# actually pulling the data from the DB is the most expensive step so we don't want to run it every time
# we make use of streamlit's cache function to avoid repeating this
# we need to include the host as a parameter so that the system re-queries when the host changes
@st.cache()
def get_quick_progress(timeframe, host):
	quick_view, summary_df = udash.get_quick_view_progress(timeframe=timeframe, drop_test=False)
	return quick_view, summary_df

@st.cache
def get_failed(timeframe, host):
	failed_visit_exams, html = udash.get_failed_visit_exams(timeframe=timeframe)
	return failed_visit_exams

if success_or_failed == 'Successful':
	with st.spinner('Querying successful visits'):
		quick_df, summary_df = get_quick_progress(timeframe, host)
	df_copy = copy.deepcopy(quick_df)

else:
	with st.spinner('Querying failed visits'):
		failed_df = get_failed(timeframe, host)
	df_copy = copy.deepcopy(failed_df)
		



# filter based on date range and specific location
if start_date > end_date:
	st.error('Error: End date must fall after start date.')
elif start_date == end_date:
	df_copy = df_copy.loc[(df_copy['datetime_EST'] > start_date)]
else:
	df_copy = df_copy.loc[(df_copy['datetime_EST'] > start_date) & (df_copy['datetime_EST'] < end_date)]

locations = st.sidebar.multiselect(
							'Locations',
							np.unique(df_copy['location']))

if len(locations) > 0:
	df_copy = df_copy.loc[df_copy['location'].isin(locations)]

if success_or_failed == 'Successful':
	quick_view = df_copy
	# -------------------------------------------------------------------------------------------------------
	# do a bunch of calculations on the quick_view so that the table is nicer to look at
	quick_view['success'] = (quick_view['need_to_check'] == False) & (quick_view['processing_attempted'] > 0)
	quick_view['data_upload_success'] = (quick_view['data_uploaded'] == quick_view['visit_exam_count'])
	quick_view['processing_success'] = (quick_view['processing_attempted'] == quick_view['visit_exam_count']) & (quick_view['processing_failed'] == 0) & (quick_view['data_upload_success'] == 1)
	quick_view['time'] = quick_view['datetime_EST'].dt.time
	quick_view['date'] = quick_view['datetime_EST'].dt.date

	diagnostic_dict = dict()
	diagnostic_dict['Number of system uses'] = str(quick_view.shape[0])
	diagnostic_dict['Full data upload'] = str(len(quick_view.loc[(quick_view['data_missing'] == False) & (quick_view['data_uploaded'] > 0)])) + ' / ' + str(quick_view.shape[0]) + ' visit(s)'
	diagnostic_dict['Data missing'] = str(int(quick_view['data_missing'].sum())) + ' / ' + diagnostic_dict['Number of system uses'] + ' visit(s)'
	diagnostic_dict['Number of digital exams'] = str(int(quick_view['visit_exam_count'].sum()))
	diagnostic_dict['Exams w/Data'] = str(int(quick_view['data_uploaded'].sum())) + ' digital exam(s)'
	diagnostic_dict['Processing Failed'] = str(int(quick_view['processing_failed'].sum())) + ' / ' + str(int(quick_view['processing_attempted'].sum())) + ' digital exam(s)'


	diagnostic_dict['Successful End-to-End'] = str(int(quick_view['success'].sum())) + ' / ' + diagnostic_dict['Number of system uses'] + ' visit(s)'
	diagnostic_df = pd.DataFrame.from_dict(diagnostic_dict, orient='index')

	# -------------------------------------------------------------------------------------------------------

	# actually print out the data and associated stats
	st.subheader("General Usage Statistics For Selected Paramters")
	st.write(diagnostic_dict)

	view = quick_view
	view = view.drop(['data_uploaded', 'processing_attempted', 'processing_failed', 'num_unprocessed', 'need_to_check', 'missing_visit_exams'], axis=1)
	order = ['visit_id', 'date', 'time', 'location', 'subject', 'visit_exam_count', 'data_upload_success', 'processing_success', 'success']
	view = view.loc[view['success'] == True]
	view = view[order]
	view[['data_upload_success', 'processing_success', 'success']] = view[['data_upload_success', 'processing_success', 'success']].replace({0: 'False', 1: 'True'})

	date_group = view.groupby(['date']).size()
	# st.write(type(date_group))
	st.subheader("Filtered Data by Visit")
	view

	# -------------------------------------------------------------------------------------------------------

	# generate charts/visualizations of data
	# st.area_chart(date_group, height=200, use_container_width=True)

	if (end_date - start_date).days > 1:
		n_unique_dates = len(date_group)

		fig = plt.figure(figsize=(10,2))

		ax = sns.barplot(x=date_group.index, y=date_group)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_xticklabels(labels=date_group.index, rotation=70, rotation_mode="anchor", ha="right")
		st.subheader("Visits per day")
		st.pyplot(fig, height=900)
	else:
		fig = plt.figure(figsize=(10,2))

		ax = sns.lineplot(x=view['time'], y=view['visit_exam_count'])

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.set_xlabel("Time Taken")
		ax.set_ylabel("Visit Exams")
		# ax.set_xticklabels(labels=date_group.index, rotation=70, rotation_mode="anchor", ha="right")
		st.subheader("Visit Exams Throughout Day")
		st.pyplot(fig, height=900)


# -------------------------------------------------------------------------------------------------------

else:
	failed_visit_exams = df_copy
	columns = ['datetime_EST', 'visit_id', 'exam', 'exam_version', 'reports_processed', 'has_error',
			'error_description', 'device_id', 'subject_id', 'visit_exam_id', 'location', 'subject', 'csv_on_s3',
			'csv_size_mb', 'audio_on_s3', 'wav_size_mb']
	failed_visit_exams = failed_visit_exams[columns]

	st.subheader("Failed visit exams by location")
	st.write(failed_visit_exams['location'].value_counts())

	st.subheader("Failed visit exams by exam and version")
	st.write(pd.crosstab(failed_visit_exams.exam,failed_visit_exams.exam_version))
	st.write(failed_visit_exams['error_description'].value_counts())

	st.subheader("Filtered failed visit exams")
	st.write(failed_visit_exams)\









