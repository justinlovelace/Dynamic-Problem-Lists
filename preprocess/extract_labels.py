from os.path import join
import pandas as pd
import yaml
import numpy as np
import sys
from tqdm import tqdm
from os.path import join
import clean_text

def preprocess_all_notes(config):
    data_dir = config['data_dir']

    # get relevant data from NOTES table
    print('\nImporting data from NOTEEVENTS...')
    path_notes = join(data_dir, 'NOTEEVENTS.csv')
    df_notes = pd.read_csv(path_notes)

    df_notes = df_notes[(df_notes['ISERROR'] != 1)]
    df_notes = df_notes[
        ['ROW_ID', 'HADM_ID', 'SUBJECT_ID', 'CHARTDATE', 'CHARTTIME', 'TEXT', 'CATEGORY']]

    clean_notes_all = clean_text.preprocess(df_notes)
    local_dir = config['local_data']
    datasetPath = join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    clean_notes_all.to_csv(datasetPath, index=False)


def get_notes(config, clean_notes_all):
    data_dir = config['data_dir']
    local_dir = config['local_data']

    clean_notes_all['CHARTDATE'] = clean_notes_all['CHARTDATE'].astype('datetime64[ns]')
    clean_notes_all['CHARTTIME'] = clean_notes_all['CHARTTIME'].astype('datetime64[ns]')
    print(clean_notes_all.dtypes)

    print('\nImporting data from ICUSTAYS...')
    path_icu = join(data_dir, 'ICUSTAYS.csv')
    df_icu = pd.read_csv(path_icu)
    df_icu = df_icu[['ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID', 'INTIME', 'OUTTIME']]
    df_icu['INTIME'] = df_icu['INTIME'].astype('datetime64[ns]')
    df_icu['OUTTIME'] = df_icu['OUTTIME'].astype('datetime64[ns]')
    print(df_icu.dtypes)

    print('\nDropping ICUSTAYS with missing times...')
    df_icu = df_icu[df_icu.OUTTIME.isnull() == False]
    df_icu = df_icu[df_icu.INTIME.isnull() == False]

    print('\nImporting data from ADMISSIONS...')
    path_adm = join(data_dir, 'ADMISSIONS.csv')
    df_adm = pd.read_csv(path_adm)
    df_adm = df_adm[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DISCHARGE_LOCATION', 'HOSPITAL_EXPIRE_FLAG', 'DEATHTIME']]
    df_adm['HOSPITAL_EXPIRE_FLAG'] = df_adm['HOSPITAL_EXPIRE_FLAG'].astype('bool')
    print(df_adm.dtypes)

    print('\nImporting data from PATIENTS...')
    path_patients = join(data_dir, 'PATIENTS.csv')
    df_patients = pd.read_csv(path_patients)
    df_patients = df_patients[['SUBJECT_ID', 'DOB', 'DOD', 'EXPIRE_FLAG']]
    df_patients['EXPIRE_FLAG'] = df_patients['EXPIRE_FLAG'].astype('bool')
    print(df_patients.dtypes)

    print('\nMerging ADMISSIONS and PATIENTS...')
    df_adm_pt = pd.merge(df_adm, df_patients, how='inner', on=['SUBJECT_ID'])
    ages = (df_adm_pt['ADMITTIME'].astype('datetime64[ns]') - df_adm_pt['DOB'].astype('datetime64[ns]')).dt.days / 365
    df_adm_pt['AGE'] = [age if age >= 0 else 91.4 for age in ages]
    df_adm_pt.drop(['DOB'], axis=1, inplace=True)
    # Removing minors from the data
    num_adm = len(df_adm_pt)
    df_adm_pt = df_adm_pt[(df_adm_pt['AGE'] >= 18)]
    df_adm_pt.drop(['AGE'], axis=1, inplace=True)
    print('Dropped ' + str(num_adm - len(df_adm_pt)) + ' minors')

    print('Merging ICUSTAYS with ADMISSIONS and PATIENTS...')
    df_icu_adm_pt = pd.merge(df_icu, df_adm_pt, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
    print(df_icu_adm_pt.dtypes)

    print('\nMerging ICUSTAYS and Preprocessed Notes...')
    df_icu_notes = pd.merge(df_icu_adm_pt, clean_notes_all, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
    print(df_icu_notes.dtypes)

    df_icu_notes = df_icu_notes[(df_icu_notes.CHARTDATE < df_icu_notes.OUTTIME.astype('datetime64[D]')) | (df_icu_notes.CHARTTIME < df_icu_notes.OUTTIME)]
    print('Total number of notes: ' + str(len(df_icu_notes)))

    clean_notes = df_icu_notes


    datasetPath = join(local_dir, 'df_MASTER_NOTES_ICU.csv')
    clean_notes.to_csv(datasetPath, index=False)

    print("\nCalculating counts of notes...")
    df_icu_notes_count = df_icu_notes.groupby('ICUSTAY_ID')['TEXT'].size().reset_index(name='counts')
    # print(df_icu_notes_count)
    df_icu_notes_count = df_icu_notes_count[(df_icu_notes_count['counts'] > 2)]


    print('\nConcatenating notes...')
    clean_notes.sort_values(['ICUSTAY_ID', 'CHARTDATE'], ascending=[True, True])
    clean_notes = clean_notes.groupby('ICUSTAY_ID')['TEXT'].apply(lambda x: '\n'.join(map(str, x)))
    clean_notes = clean_notes.reset_index()
    print('Total number of icustays: ' + str(len(clean_notes)))

    print("\nDropping stays with less than 3 notes...")
    clean_notes = pd.merge(clean_notes, df_icu_notes_count, how='inner', on=['ICUSTAY_ID'])
    clean_notes = clean_notes[['ICUSTAY_ID', 'TEXT']]
    clean_notes = pd.merge(clean_notes, df_icu_adm_pt, how='inner', on=['ICUSTAY_ID'])
    print(clean_notes.dtypes)
    clean_notes = clean_notes[['ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID', 'INTIME', 'OUTTIME', 'ADMITTIME', 'DISCHTIME', 'DISCHARGE_LOCATION', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG', 'DOD', 'EXPIRE_FLAG', 'TEXT']]
    print('Total number of icustays: ' + str(len(clean_notes)))

    return clean_notes

def add_features(config, df_MASTER_DATA):
    """
    the function adds target labels to the dataset
    :param df_MASTER_DATA:
    :return:
    """

    data_dir = config['data_dir']
    print('READING ICD CODES...')
    path_proc_icd = join(data_dir, 'PROCEDURES_ICD.csv')
    path_diag_icd = join(data_dir, 'DIAGNOSES_ICD.csv')

    df_proc_icd = pd.read_csv(path_proc_icd, dtype={'ICD9_CODE': 'object'})
    df_diag_icd = pd.read_csv(path_diag_icd, dtype={'ICD9_CODE': 'object'})

    print('READING PHECODES...')
    local_dir = config['local_data']
    path_phecodes = join(local_dir, 'phecode_icd9_rolled.csv')
    df_phecodes = pd.read_csv(path_phecodes, dtype={'PheCode': 'object'})
    df_phecodes['PheCode'] = df_phecodes['PheCode'].apply(lambda x: x.replace('.', ''))

    df_phecodes['ICD9'] = df_phecodes['ICD9'].apply(lambda x: x.replace('.', ''))
    df_phecodes = df_phecodes[['ICD9', 'PheCode']]
    df_phecodes.rename(columns={"ICD9": "ICD9_CODE"}, inplace=True)
    df_phecodes['ROLLED_PHECODE'] = df_phecodes['PheCode'].apply(lambda x: x[0:3])
    phecode_column_names = ['PHEC_' + x for x in df_phecodes.PheCode.unique()]
    rolled_phecode_column_names = ['ROLLED_PHEC_' + x for x in df_phecodes.ROLLED_PHECODE.unique()]
    for name in phecode_column_names:
        assert len(name) >= 8
        df_MASTER_DATA[name] = 0
    for name in rolled_phecode_column_names:
        assert len(name) >= 15
        df_MASTER_DATA[name] = 0
    print('Added ' + str(len(phecode_column_names)) + ' phecode columns and ' + str(len(rolled_phecode_column_names)) + ' rolled phecode columns')

    print('MATCHING PHECODES...')
    print(df_diag_icd['ICD9_CODE'].head(10))
    print(df_diag_icd.dtypes)
    df_diag_icd = pd.merge(df_diag_icd, df_phecodes, how='left', on='ICD9_CODE')
    print(df_diag_icd.dtypes)


    print('ADDING ICD COLUMNS TO DATAFRAME...')

    proc_column_names = ['ICD_PROC_' + str(x) for x in df_proc_icd.ICD9_CODE.unique()]
    diag_column_names = ['ICD_DIAG_' + str(x) for x in df_diag_icd.ICD9_CODE.unique()]
    for name in proc_column_names:
        assert len(name) >= 12
        df_MASTER_DATA[name] = 0
    for name in diag_column_names:
        assert len(name) >= 12
        df_MASTER_DATA[name] = 0
    print('Added ' + str(len(proc_column_names) + len(diag_column_names)) + 'icd columns')

    df_proc_icd['ROLLED_ICD9_CODE'] = df_proc_icd['ICD9_CODE'].apply(lambda x: str(x)[0:3])
    df_diag_icd['ROLLED_ICD9_CODE'] = df_diag_icd['ICD9_CODE'].apply(lambda x: str(x)[0:3])
    rolled_proc_column_names = ['ROLLED_ICD_PROC_' + str(x) for x in df_proc_icd.ROLLED_ICD9_CODE.unique()]
    rolled_diag_column_names = ['ROLLED_ICD_DIAG_' + str(x) for x in df_diag_icd.ROLLED_ICD9_CODE.unique()]
    print('Added ' + str(len(rolled_proc_column_names) + len(rolled_diag_column_names)) + 'rolled icd columns')

    for name in rolled_proc_column_names:
        assert len(name) >= 19
        df_MASTER_DATA[name] = 0
    for name in rolled_diag_column_names:
        assert len(name) >= 19
        df_MASTER_DATA[name] = 0

    print('\n Adding target variables...')

    df_MASTER_DATA['INTIME'] = df_MASTER_DATA['INTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['OUTTIME'] = df_MASTER_DATA['OUTTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['ADMITTIME'] = df_MASTER_DATA['ADMITTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DISCHTIME'] = df_MASTER_DATA['DISCHTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DEATHTIME'] = df_MASTER_DATA['DEATHTIME'].astype('datetime64[ns]')
    df_MASTER_DATA['DOD'] = df_MASTER_DATA['DOD'].astype('datetime64[ns]')
    df_MASTER_DATA = df_MASTER_DATA.sort_values(['SUBJECT_ID', 'INTIME', 'OUTTIME'], ascending=[True, True, True])
    df_MASTER_DATA.reset_index(inplace=True, drop=True)

    # Add targetr column to show if readmitted within different timeframes
    df_MASTER_DATA = df_MASTER_DATA.assign(Mortality_30days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(Mortality_InHospital=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_30days=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(IsReadmitted_Bounceback=0)
    df_MASTER_DATA = df_MASTER_DATA.assign(Time_To_readmission=np.nan)
    df_MASTER_DATA = df_MASTER_DATA.assign(Time_To_death=np.nan)

    # total number of admissions
    num_adms = df_MASTER_DATA.shape[0]
    indexes_to_drop = []
    for idx in tqdm(range(0, num_adms)):
        # Add all ICD labels
        icd_proc_labels = df_proc_icd.loc[df_proc_icd['HADM_ID'] == df_MASTER_DATA.HADM_ID[idx]]
        for index, row in icd_proc_labels.iterrows():
            assert 'ICD_PROC_' + str(row['ICD9_CODE']) in df_MASTER_DATA.columns
            assert 'ROLLED_ICD_PROC_' + str(row['ROLLED_ICD9_CODE']) in df_MASTER_DATA.columns
            df_MASTER_DATA.loc[idx, 'ICD_PROC_' + str(row['ICD9_CODE'])] = 1
            df_MASTER_DATA.loc[idx, 'ROLLED_ICD_PROC_' + str(row['ROLLED_ICD9_CODE'])] = 1
        icd_diag_labels = df_diag_icd.loc[df_diag_icd['HADM_ID'] == df_MASTER_DATA.HADM_ID[idx]]
        for index, row in icd_diag_labels.iterrows():
            assert 'ICD_DIAG_' + str(row['ICD9_CODE']) in df_MASTER_DATA.columns
            assert 'ROLLED_ICD_DIAG_' + str(row['ROLLED_ICD9_CODE']) in df_MASTER_DATA.columns
            if 'PHEC_' + str(row['PheCode']) in df_MASTER_DATA.columns:
                df_MASTER_DATA.loc[idx, 'PHEC_' + str(row['PheCode'])] = 1
                df_MASTER_DATA.loc[idx, 'ROLLED_PHEC_' + str(row['ROLLED_PHECODE'])] = 1
            df_MASTER_DATA.loc[idx, 'ICD_DIAG_' + str(row['ICD9_CODE'])] = 1
            df_MASTER_DATA.loc[idx, 'ROLLED_ICD_DIAG_' + str(row['ROLLED_ICD9_CODE'])] = 1

        # Drops icustay from cohort if the patient dies during the stay
        if df_MASTER_DATA.HOSPITAL_EXPIRE_FLAG[idx] and (df_MASTER_DATA.DEATHTIME[idx] <= df_MASTER_DATA.OUTTIME[idx] or
                                                  (df_MASTER_DATA.DISCHTIME[idx] <= df_MASTER_DATA.OUTTIME[idx] and df_MASTER_DATA.DISCHARGE_LOCATION[idx] == 'DEAD/EXPIRED')):
            indexes_to_drop.append(idx)
        # Calculates readmissions labels
        if idx > 0 and df_MASTER_DATA.SUBJECT_ID[idx] == df_MASTER_DATA.SUBJECT_ID[idx - 1]:
            # previous icu discharge time
            prev_outtime = df_MASTER_DATA.OUTTIME[idx - 1]
            # current icu admit time
            curr_intime = df_MASTER_DATA.INTIME[idx]

            readmit_time = curr_intime - prev_outtime
            df_MASTER_DATA.loc[idx - 1, 'Time_To_readmission'] = readmit_time.seconds / (3600 * 24) + readmit_time.days

            if readmit_time.days <= 30:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_30days'] = 1
                # Check bouncebacks
            if df_MASTER_DATA.HADM_ID[idx] == df_MASTER_DATA.HADM_ID[idx - 1]:
                df_MASTER_DATA.loc[idx - 1, 'IsReadmitted_Bounceback'] = 1

        # Checks in hospital mortality
        if df_MASTER_DATA.HOSPITAL_EXPIRE_FLAG[idx]:
            df_MASTER_DATA.loc[idx, 'Mortality_InHospital'] = 1
            # current icu discharge time
            outtime = df_MASTER_DATA.OUTTIME[idx]
            # current death time
            deathtime = df_MASTER_DATA.DEATHTIME[idx]
            time_to_death = deathtime - outtime
            df_MASTER_DATA.loc[idx, 'Time_To_death'] = time_to_death.seconds / (3600 * 24) + time_to_death.days
            if time_to_death.days <= 30:
                df_MASTER_DATA.loc[idx, 'Mortality_30days'] = 1

        # Checks out of hospital mortality
        if df_MASTER_DATA.EXPIRE_FLAG[idx]:
            # current icu discharge time
            outtime = df_MASTER_DATA.OUTTIME[idx]
            # current death time
            dod = df_MASTER_DATA.DOD[idx]
            time_to_death = dod - outtime
            df_MASTER_DATA.loc[idx, 'Time_To_death'] = time_to_death.seconds / (3600 * 24) + time_to_death.days
            if time_to_death.days <= 30:
                df_MASTER_DATA.loc[idx, 'Mortality_30days'] = 1
    df_MASTER_DATA.drop(df_MASTER_DATA.index[indexes_to_drop], inplace=True)
    print('Total stays: ' + str(len(df_MASTER_DATA)))
    print('30 day readmission: ' + str(df_MASTER_DATA['IsReadmitted_30days'].sum()))
    print('Bounceback readmission: ' + str(df_MASTER_DATA['IsReadmitted_Bounceback'].sum()))
    print('30 day mortality: ' + str(df_MASTER_DATA['Mortality_30days'].sum()))
    print('In hospital mortality: ' + str(df_MASTER_DATA['Mortality_InHospital'].sum()))
    return df_MASTER_DATA


if __name__ == "__main__":
    config = yaml.safe_load(open("../resources/config.yml"))
    local_dir = config['local_data']

    preprocess_all_notes(config)

    print('\nImporting data from CNN preprocessed notes...')
    path_clean_notes = join(local_dir, 'df_MASTER_NOTES_ALL.csv')
    clean_notes_all = pd.read_csv(path_clean_notes)
    notes = get_notes(config, clean_notes_all)

    master_notes = add_features(config, notes)
    print(master_notes[:5]['TEXT'])
    datasetPath = join(local_dir, 'df_MASTER_DATA_ALL_LABELS.csv')
    master_notes.to_csv(datasetPath, index=False)

    columns = ['ICUSTAY_ID']
    for i in range(5):
        print('READING TRAINING DATA FOR FOLD ' + str(i))
        df_train = pd.read_csv(join(local_dir, 'fold' + str(i), 'df_train_subjects.csv'))
        df_train = df_train[columns]
        print('READING VALIDATION DATA FOR FOLD ' + str(i))
        df_val = pd.read_csv(join(local_dir, 'fold' + str(i), 'df_val_subjects.csv'))
        df_val = df_val[columns]
        print('READING TEST DATA FOR FOLD ' + str(i))
        df_test = pd.read_csv(join(local_dir, 'fold' + str(i), 'df_test_subjects.csv'))
        df_test = df_test[columns]


        print('Merging train...')
        df_upd_train = pd.merge(master_notes, df_train, how='inner', on='ICUSTAY_ID')
        print('Merging val...')
        df_upd_val = pd.merge(master_notes, df_val, how='inner', on='ICUSTAY_ID')
        print('Merging test...')
        df_upd_test = pd.merge(master_notes, df_test, how='inner', on='ICUSTAY_ID')

        datasetPath = join(local_dir, 'fold' + str(i), 'df_train_subjects.csv')
        df_upd_train.to_csv(datasetPath, index=False)

        datasetPath = join(local_dir, 'fold' + str(i), 'df_val_subjects.csv')
        df_upd_val.to_csv(datasetPath, index=False)

        datasetPath = join(local_dir, 'fold' + str(i), 'df_test_subjects.csv')
        df_upd_test.to_csv(datasetPath, index=False)

