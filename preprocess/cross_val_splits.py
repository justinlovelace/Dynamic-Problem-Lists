import pandas as pd
from os.path import join
import yaml
from tabulate import tabulate
from sklearn.model_selection import KFold
import sys

def split_data(config, df_MASTER):
    df_SUBJECT_ID = df_MASTER[['SUBJECT_ID']].drop_duplicates()
    df_SUBJECT_ID.sort_values(['SUBJECT_ID'], ascending=[True], inplace=True)  # make sure that the data has a fixed order before shuffling

    print('Generating Splits...')
    kf = KFold(n_splits=5, random_state=100, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(df_SUBJECT_ID):
        train_subjects = df_SUBJECT_ID.iloc[train_index]
        test_subjects = df_SUBJECT_ID.iloc[test_index]
        dev_subjects = test_subjects.sample(frac=0.5, random_state=100)
        test_subjects = test_subjects.drop(dev_subjects.index)
        df_TEST = df_MASTER[(df_MASTER.SUBJECT_ID.isin(test_subjects.SUBJECT_ID))]
        df_DEV = df_MASTER[(df_MASTER.SUBJECT_ID.isin(dev_subjects.SUBJECT_ID))]
        df_TRAIN = df_MASTER[(df_MASTER.SUBJECT_ID.isin(train_subjects.SUBJECT_ID))]
        print('-----------------' + 'FOLD ' + str(fold) + '-----------------')
        print('Split Stats: ')
        total = len(df_MASTER.index)
        test_count = len(df_TEST.index)
        dev_count = len(df_DEV.index)
        train_count = len(df_TRAIN.index)

        print(tabulate(
            [['Total', str(total), df_MASTER['IsReadmitted_30days'].sum(), df_MASTER['IsReadmitted_Bounceback'].sum(), df_MASTER['Mortality_30days'].sum()],
             ['Training', str(train_count), df_TRAIN['IsReadmitted_30days'].sum(), df_TRAIN['IsReadmitted_Bounceback'].sum(), df_TRAIN['Mortality_30days'].sum()],
             ['Dev', str(dev_count), df_DEV['IsReadmitted_30days'].sum(), df_DEV['IsReadmitted_Bounceback'].sum(), df_DEV['Mortality_30days'].sum()],
             ['Test', str(test_count), df_TEST['IsReadmitted_30days'].sum(), df_TEST['IsReadmitted_Bounceback'].sum(), df_TEST['Mortality_30days'].sum()]],
            headers=['Set', 'ICUSTAYS', 'IsReadmitted_30days', 'IsReadmitted_Bounceback', 'Mortality_30days']))
        print('\n')
        print(tabulate(
            [['Total', str(total), df_MASTER['IsReadmitted_30days'].mean(), df_MASTER['IsReadmitted_Bounceback'].mean(), df_MASTER['Mortality_30days'].mean()],
             ['Training', str(train_count), df_TRAIN['IsReadmitted_30days'].mean(), df_TRAIN['IsReadmitted_Bounceback'].mean(), df_TRAIN['Mortality_30days'].mean()],
             ['Dev', str(dev_count), df_DEV['IsReadmitted_30days'].mean(), df_DEV['IsReadmitted_Bounceback'].mean(), df_DEV['Mortality_30days'].mean()],
             ['Test', str(test_count), df_TEST['IsReadmitted_30days'].mean(),df_TEST['IsReadmitted_Bounceback'].mean(), df_TEST['Mortality_30days'].mean()]],
            headers=['Set', 'ICUSTAYS', 'IsReadmitted_30days', 'IsReadmitted_Bounceback', 'Mortality_30days']))

        local_dir = config['local_data']
        datasetPath = join(local_dir, 'fold'+str(fold), 'df_train_subjects.csv')
        print(datasetPath)
        df_TRAIN.to_csv(datasetPath, index=False)
        datasetPath = join(local_dir, 'fold'+str(fold), 'df_val_subjects.csv')
        print(datasetPath)
        df_DEV.to_csv(datasetPath, index=False)
        datasetPath = join(local_dir, 'fold'+str(fold), 'df_test_subjects.csv')
        print(datasetPath)
        df_TEST.to_csv(datasetPath, index=False)
        fold += 1

def split_bert(config, df_bert):
    local_dir = config['local_data']
    for fold in range(5):
        print('-----------------' + 'FOLD ' + str(fold) + '-----------------')
        print('LOADINING SPLITS...')
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_train_subjects.csv')
        print(datasetPath)
        train_subjects = pd.read_csv(datasetPath)
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_val_subjects.csv')
        print(datasetPath)
        dev_subjects = pd.read_csv(datasetPath)
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_test_subjects.csv')
        print(datasetPath)
        test_subjects = pd.read_csv(datasetPath)
        print('CALCULATING SPLITS...')
        df_TEST = df_bert[(df_bert.SUBJECT_ID.isin(test_subjects.SUBJECT_ID))]
        df_DEV = df_bert[(df_bert.SUBJECT_ID.isin(dev_subjects.SUBJECT_ID))]
        df_TRAIN = df_bert[(df_bert.SUBJECT_ID.isin(train_subjects.SUBJECT_ID))]
        print('SAVING SPLITS...')
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_train_subjects_bert.csv')
        print(datasetPath)
        df_TRAIN.to_csv(datasetPath, index=False)
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_val_subjects_bert.csv')
        print(datasetPath)
        df_DEV.to_csv(datasetPath, index=False)
        datasetPath = join(local_dir, 'fold' + str(fold), 'df_test_subjects_bert.csv')
        print(datasetPath)
        df_TEST.to_csv(datasetPath, index=False)

if __name__ == "__main__":
    config = yaml.safe_load(open("../resources/config.yml"))
    local_dir = config['local_data']
    path_notes = join(local_dir, 'df_MASTER_DATA_ALL_LABELS.csv')
    print("Loading note data...")
    df_notes = pd.read_csv(path_notes)
    # Create cross-val splits
    split_data(config, df_notes)

    # Match cross-val splits for bert
    path_notes = join(local_dir, 'df_MASTER_DATA_ALL_LABELS_BERT.csv')
    print("Loading Bert note data...")
    df_bert = pd.read_csv(path_notes)
    split_bert(config, df_bert)
