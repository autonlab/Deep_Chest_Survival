import pandas as pd
import numpy as np


def _encode_cols_index(df):

	columns = df.columns
	# Convert Objects to Strings
	for col in columns:
		if df[col].dtype == 'O':
			df.loc[:, col] = df[col].values.astype(str)

	# If Index is Object, covert to String
	if df.index.dtype == 'O':
	 	df.index = df.index.values.astype(str)

	return df


def _load_generic_biolincc_dataset(outcome_tbl, time_col, event_col, features,
									id_col, visit_col=None, baseline_visit=None,
									location=''):

	if not isinstance(baseline_visit, (tuple, set, list)):
		baseline_visit = [baseline_visit]

	# List of all features to extract
	all_features = []
	for feature in features:
		all_features+=features[feature]
	all_features = list(set(all_features)) # Only take the unqiue columns

	if '.sas' in outcome_tbl: outcomes = pd.read_sas(location+outcome_tbl, index=id_col, format = 'sas7bdat', encoding="latin-1")
	elif '.csv' in outcome_tbl: outcomes = pd.read_csv(location+outcome_tbl, index_col=id_col, encoding='latin-1')
	else: raise NotImplementedError()

	outcomes = outcomes[[time_col, event_col]]

	dataset = outcomes.copy()
	dataset.columns = ['time', 'event']

	for feature in features:

		if '.sas' in outcome_tbl: table = pd.read_sas(location+feature, index=id_col, format = 'sas7bdat', encoding="latin-1")
		elif '.csv' in outcome_tbl: table = pd.read_csv(location+feature, index_col=id_col)
		else: raise NotImplementedError()

		if (visit_col is not None) and (visit_col in table.columns):
			mask = np.zeros(len(table[visit_col])).astype('bool')
			for baseline_visit_ in baseline_visit:
				mask = mask | (table[visit_col]==baseline_visit_)
			table = table[mask]
		table = table[features[feature]]
		print(table.shape)
		dataset = dataset.join(table)

	outcomes = dataset[['time', 'event']]
	features = dataset[all_features]
	# for race, only leave white and black there, and everything else is 'other'
	mask = ~((features.race7 == 1) | (features.race7 == 2))
	features.race7[mask] = 3

	outcomes = _encode_cols_index(outcomes)
	features = _encode_cols_index(features)

	return outcomes, features


def _get_images(location):

	path = location+'Standard 25K Linkage (2021)/link_2021_25k_selection.sas7bdat'
	if '.sas' in path: images = pd.read_sas(path, format = 'sas7bdat', encoding="latin-1")

	images['image_file_name'] = images['image_file_name'].apply(lambda x: x[:2].upper()+x[2:])

	images.drop('assoc_visit_visnum', axis=1, inplace=True)
	images = images.set_index(['plco_id', 'assoc_visit_syr'], append=False)

	return images


def _get_folds(y_tr, k):

	patient_ids = y_tr.index.get_level_values(0).unique()
	patient_ids = pd.DataFrame({'patient_ids': patient_ids.values},
								index=patient_ids)

	folds = (list(range(k))*len(patient_ids))[:len(patient_ids)]

	patient_ids['folds'] = folds

	patient_folds = pd.DataFrame(index=y_tr.index)
	patient_folds['patient_ids'] = patient_folds.index.get_level_values(0).values
	patient_folds['screen'] = patient_folds.index.get_level_values(1).values

	patient_folds = patient_folds.merge(patient_ids,
										how='left',
										left_on='patient_ids',
										right_on='patient_ids')

	return patient_folds


def load_plco_trial_data(location='datasets/package-plco/Lung/'):

	feature_list = {
		# XRY findings
		'Screening/lung_screen_data_nov18_d070819.sas7bdat' : ['as_mass', 'as_mass_cnt', 'as_mass_loc', 'as_mass_loc_pos', 'as_mass_loc_side',
																'as_nodule', 'as_nodule_cnt', 'as_nodule_loc', 'as_nodule_loc_pos', 'as_nodule_loc_side',
																'as_other', 'as_other_cnt', 'as_other_desc', 'as_other_loc', 'as_other_loc_pos', 'as_other_loc_side',
																'an_bone', 'an_cardiac', 'an_copd', 'an_gran', 'an_pleufibro', 'an_pleufluid', 'an_scar',
																'as_atelect', 'as_atelect_cnt', 'as_atelect_loc', 'as_atelect_loc_pos', 'as_atelect_loc_side',
																'as_hilar', 'as_hilar_cnt', 'as_hilar_loc', 'as_hilar_loc_pos', 'as_hilar_loc_side',
																'as_infiltrate', 'as_infiltrate_cnt', 'as_infiltrate_loc', 'as_infiltrate_loc_pos', 'as_infiltrate_loc_side',
																'as_pleural', 'as_pleural_cnt', 'as_pleural_loc', 'as_pleural_loc_pos', 'as_pleural_loc_side',
																'radiographic_abnorm',
																'study_yr', 'xry_days', 'xry_result'],
		# demographic data
		'lung_data_nov18_d070819.sas7bdat' : ['age', 'agelevel', 'center', 'rndyear', 'sex', 'dual', 'race7', 'hispanic_f', 'cig_stat', 'lung_cancer']
	}

	outcomes, features = None, None
	for study_yr in [0.0, 1.0, 2.0, 3.0]:
	# for study_yr in [3.0]:
		print(study_yr)
		outcomes_, features_ = _load_generic_biolincc_dataset('lung_data_nov18_d070819.sas7bdat',
																'mortality_exitdays', 'mortality_exitstat',
																features=feature_list,
																id_col='plco_id',
																visit_col='study_yr',
																baseline_visit=study_yr,
																location=location)
		# Drop all individuals not corresponding to the study_yr
		outcomes_['study_yr'] = np.array([study_yr]*len(outcomes_))
		outcomes_ = outcomes_[features_.study_yr == study_yr].set_index(['study_yr'], append=True)
		features_ = features_[features_.study_yr == study_yr].set_index(['study_yr'], append=True)

		if outcomes is None:
			outcomes = outcomes_
			features = features_
		else:
			outcomes = pd.concat((outcomes, outcomes_))
			features = pd.concat((features, features_))

	# Sort the features/outcomes by PLCO ID and Study ID
	features.sort_index(inplace=True)
	outcomes.sort_index(inplace=True)

	# Calculate the Days from XRY to the Mortality_exit_days
	outcomes['time'] = outcomes['time'] - features['xry_days']
	# Add years to the age at randomization to get age at screening time
	features['age'] = features['age'] + features['xry_days']/365.25

	# Drop the xry_days columns that is not needed
	features.drop('xry_days', axis=1, inplace=True)

	# Drop the inadequate scans.
	outcomes = outcomes[features.xry_result != 4.0]
	features = features[features.xry_result != 4.0]
	features.drop('xry_result', axis=1, inplace=True)

	# Drop the scans that are absurd with negative times
	features = features[outcomes.time > 0.0]
	outcomes = outcomes[outcomes.time > 0.0]

	all_data = outcomes.join(features, how='left')
	all_data.index.set_names(['plco_id', 'assoc_visit_syr'], inplace=True)

	# all models
	# train on entire dataset*0.6, test entire dataset*0.2, val entire *0.2
	# train on entire dataset*0.6, test study year=3*0.2, val study year=3*0.2
	# train on study year=3*0.6, test on study year=3*0.2, val study year=3*0.2
	# train on study year=3*0.6, test entire dataset*0.2, val entire *0.2

	all_data['cv_folds'] = all_data.index.get_level_values(0).map(lambda x : int(x[-1])%5)
	# get image names
	images = _get_images(location)

	# Lower case all image names
	images.image_file_name = images.image_file_name.apply(lambda x: x.lower())
	# Lower case the batch names
	images.batch_number = images.batch_number.apply(lambda x: x.lower())


	all_data = images.join(all_data, how='left')

	all_data = all_data.reset_index().set_index(['image_file_name', 'batch_number'], append=False)

	# Convert Censoring Indicator to Binary
	# ie. 1 = Event, 0 = Censored
	all_data.event = (all_data.event == 1.0)
	# Drop the scans that are absurd with negative times
	all_data = all_data[all_data.time > 0.0]

	return all_data



