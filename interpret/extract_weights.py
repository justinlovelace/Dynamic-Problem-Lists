"""Evaluates the model"""

import argparse
import os

import numpy as np
import utils
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='/', help="Directory containing params.json")

def consolidate_weights(model_dirs, params):
    dfs = []
    for model_dir in model_dirs:
        datasetPath = os.path.join(model_dir, 'df_weights.csv')
        df = pd.read_csv(datasetPath)
        dfs.append(df)

    df_master = pd.concat(dfs)
    df_agg_master = df_master.groupby(['Phenotypes'], as_index=False).agg(
        {'Weights': ['mean', 'std', 'count']})
    df_agg_master.columns = df_agg_master.columns.map('_'.join)

    print(df_agg_master.columns)
    print(df_agg_master.dtypes)
    print(df_agg_master.head(10))
    # sys.exit()
    df_agg_master.sort_values(by=['Weights_mean'], ascending=False, inplace=True)
    datasetPath = os.path.join(params.save_path, 'df_icd_weights_' + params.task + '.csv')
    df_agg_master.to_csv(datasetPath, index=False)


def extract_weights(model, data, model_dir):

    phen_weights = model.final_proj.weight.data.tolist()[0]
    print(len(phen_weights))
    print(len(data.icd_cols))
    print(model.final_proj.weight.data.size())

    df_weights = pd.DataFrame(
        {'Phenotypes': data.icd_cols, 'Weights': phen_weights})
    df_weights.sort_values(by=['Weights'], ascending=False, inplace=True)
    datasetPath = os.path.join(model_dir, 'df_weights.csv')
    df_weights.to_csv(datasetPath, index=False)

def problem_stats(params):

    label_dict = {'ROLLED_ICD_DIAG': [], 'ROLLED_ICD_PROC': [], 'ROLLED_PHEC': [], 'FULL_ICD_DIAG': [], 'FULL_ICD_PROC': [], 'FULL_PHEC': []}

    for i in range(5):
        print('LOADING FOLD '+str(i))
        datasetPath = os.path.join(params.local_data, 'fold' + str(i), 'df_train_subjects.csv')
        df_notes = pd.read_csv(datasetPath)
        label_dict['ROLLED_ICD_DIAG'].append(len([x for x in df_notes.columns if
                             'ROLLED_ICD_DIAG_' in x and df_notes[x].sum() >= 50]))
        label_dict['ROLLED_ICD_PROC'].append(len([x for x in df_notes.columns if
                             'ROLLED_ICD_PROC_' in x and df_notes[x].sum() >= 50]))
        label_dict['ROLLED_PHEC'].append(len([x for x in df_notes.columns if
                                                  'ROLLED_PHEC' in x and df_notes[x].sum() >= 50]))
        label_dict['FULL_ICD_DIAG'].append(len([x for x in df_notes.columns if
                                'ICD_DIAG' in x and 'ROLLED' not in x and df_notes[x].sum() >= 50]))
        label_dict['FULL_ICD_PROC'].append(len([x for x in df_notes.columns if
                                'ICD_PROC' in x and 'ROLLED' not in x and df_notes[x].sum() >= 50]))
        label_dict['FULL_PHEC'].append(len([x for x in df_notes.columns if
                                                'PHEC' in x and 'ROLLED' not in x and df_notes[x].sum() >= 50]))

    for key, value in label_dict.items():
        print(key)
        print(np.mean(value))

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    problem_stats(params)
    sys.exit()
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    params.model == "conv_attn_phen"
    for emb in ['w2v']:
        params.emb = emb
        params.phenotype = 'rolled_phecodes'
        for task in ['bounceback_allied']:
            params.task = task
            model_dirs = []
            for fold in range(4, 5):
                params.fold = fold

                # Set the random seed for reproducible experiments
                torch.manual_seed(230)
                np.random.seed(230)
                random.seed(230)
                if params.cuda:
                    torch.cuda.manual_seed_all(230)

                # Set the logger
                job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_{}_{}".format(params.emb, params.learning_rate,
                                                                       params.kernels,
                                                                       params.batch_size, params.filters,
                                                                       params.dropout,
                                                                       params.model, params.task)
                if 'phen' in params.model:
                    job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_bn{}_{}_icds{}>={}_icdNum{}_{}".format(params.emb,
                                                                                                    params.learning_rate,
                                                                                                    params.kernels,
                                                                                                    params.batch_size * params.grad_acc,
                                                                                                    params.filters,
                                                                                                    params.dropout,
                                                                                                    params.batch_norm,
                                                                                                    params.model,
                                                                                                    params.phenotype,
                                                                                                    params.icd_threshold,
                                                                                                    params.icd_val_threshold,
                                                                                                    params.task)
                    params.icd_model = "emb{}_k{}_f{}_{}_icds{}>={}_{}".format(params.emb,
                                                                               params.learning_rate,
                                                                               params.kernels,
                                                                               params.model,
                                                                               params.phenotype,
                                                                               params.icd_threshold,
                                                                               'icd')
                # Create a new folder in parent_dir with unique_name "job_name"
                model_dir = os.path.join(params.save_path, 'fold' + str(params.fold), job_name)
                print(model_dir)
                model_dirs.append(model_dir)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                # load data
                data_loader = DataLoader(args.data_dir, params)

                data = data_loader.load_data(['train'], args.data_dir)
                params.num_phenotypes = len(data_loader.icd_cols)

                # Define the model and optimizer
                device = torch.device("cuda:0" if params.cuda else sys.exit("gpu unavailable"))
                model = models.CNN_Text_Attn_Phen(data_loader.weights_w2v, params)

                print(model)
                utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model, parallel=False)

                extract_weights(model, data_loader, model_dir)
            consolidate_weights(model_dirs, params)