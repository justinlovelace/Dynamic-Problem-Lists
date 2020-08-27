"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.models as models
from model.data_loader import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import trange
import sys
from nltk import ngrams, pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def evaluate_attn(model, loss_fn, data_iterator, metrics, params, num_steps, data_loader, model_dir):
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    master_list = []

    # compute metrics over the dataset
    running_metrics = utils.TestMetrics()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for i in t:
            # fetch the next evaluation batch
            train_batch_w2v, labels_batch, ids = next(data_iterator)
            output_batch, attn_weights_w2v = model(train_batch_w2v, interpret=True)
            batch_word_indexes = train_batch_w2v[0].tolist()
            batch_text = []
            for word_indexes in batch_word_indexes:
                unigrams, bigrams, trigrams = [], [], []
                for ind in range(len(word_indexes)):
                    if ind < 2:
                        pre_context = data_loader.index_to_word_w2v[word_indexes[ind - 1]]
                    elif ind < 1:
                        pre_context = ''
                    else:
                        pre_context = data_loader.index_to_word_w2v[word_indexes[ind - 2]] + ' ' + \
                                      data_loader.index_to_word_w2v[word_indexes[ind - 1]]
                    if ind + 4 < len(word_indexes):
                        unigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                        bigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                        trigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 3]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 4]])
                    elif ind + 3 < len(word_indexes):
                        unigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                        bigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                        trigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 3]])
                    elif ind + 2 < len(word_indexes):
                        unigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                        bigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]])
                        trigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 2]] + '] ')
                    elif ind + 1 < len(word_indexes):
                        unigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]])
                        bigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + ' ' +
                            data_loader.index_to_word_w2v[word_indexes[ind + 1]] + '] ')
                    else:
                        unigrams.append(
                            pre_context + ' [' + data_loader.index_to_word_w2v[word_indexes[ind]] + '] ')

                batch_text.append(
                    unigrams + bigrams + ['<CONV_PAD>'] + trigrams + ['<CONV_PAD>'] + ['<CONV_PAD>'])
            output_list = output_batch.tolist()
            attn_weights_list = [x.tolist() for x in attn_weights_w2v]
            labels_batch_list = labels_batch.tolist()
            assert len(ids) == len(batch_text)
            assert len(ids) == len(labels_batch_list)
            assert len(ids) == len(output_list)
            assert len(ids) == len(attn_weights_list[0])
            for head in range(len(attn_weights_list)):
                for index in range(len(ids)):
                    temp_list = []
                    temp_list.append(ids[index])
                    temp_list.append('w2v')
                    temp_list.append(head)
                    temp_list.append(labels_batch_list[index][0])
                    temp_list.append(output_list[index][0])
                    attn_words = list(zip(attn_weights_list[head][index], batch_text[index]))
                    attn_words.sort(reverse=True)
                    new_attn_words = [x for t in attn_words[:50] for x in t]
                    temp_list.extend(new_attn_words)
                    master_list.append(temp_list)
            loss = loss_fn(output_batch, labels_batch)

            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data.item()
            summ.append(summary_batch)

    df_attn_scores = pd.DataFrame(master_list, columns=["ICUSTAY_ID", 'head', 'embedding', params.task + "_label",
                                                        params.task + "_prediction"] + [
                                                           'attn_' + str(i // 2) if i % 2 == 0 else 'words_' + str(
                                                               i // 2) for i in range(100)])
    print(df_attn_scores.dtypes)
    df_attn_scores.sort_values(by=[params.task + "_prediction"], ascending=False, inplace=True)
    print(df_attn_scores.head(5))
    datasetPath = os.path.join(model_dir, 'df_attn.csv')
    df_attn_scores.to_csv(datasetPath, index=False)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info('AUCROC' + str(running_metrics()))
    metrics = running_metrics()
    return {'AUCROC': metrics[0], "AUCPR": metrics[1]}


def evaluate_phec_attn(model, loss_fn, data_iterator, metrics, params, num_steps, data_loader, model_dir):
    # set model to evaluation mode

    model.eval()

    master_list = []

    with torch.no_grad():
        t = trange(num_steps)
        for i in t:
            # fetch the next evaluation batch
            train_batch_w2v, train_batch_sp, labels_batch, icd_labels, ids = next(data_iterator)
            output_batch, attn_weights_w2v, phen_prob, phen_contr = model(train_batch_w2v, interpret=True)
            batch_word_indexes = train_batch_w2v[0].tolist()
            batch_text = []
            for word_indexes in batch_word_indexes:
                words = [data_loader.index_to_word_w2v[x] for x in word_indexes]
                words = list(pad_sequence(words, 3, pad_left=True, left_pad_symbol='<PAD>', pad_right=True,
                                          right_pad_symbol='<PAD>'))
                words_copy = list(words)
                unigrams = list(ngrams(words_copy, 5))
                unigrams = [' '.join(x) for x in unigrams]
                bigrams = list(ngrams(words_copy, 6))
                bigrams = [' '.join(x) for x in bigrams]
                words_copy = list(words)
                trigrams = list(ngrams(words_copy, 7))
                trigrams = [' '.join(x) for x in trigrams]

                batch_text.append(
                    unigrams + bigrams + ['<CONV_PAD>'] + trigrams + ['<CONV_PAD>'] + ['<CONV_PAD>'])
            output_list = output_batch.tolist()
            attn_weights_list = [x.tolist() for x in attn_weights_w2v]
            phen_prob_list = [x.tolist() for x in phen_prob]
            phen_contr_list = [x.tolist() for x in phen_contr]
            labels_batch_list = labels_batch.tolist()
            icd_labels_list = icd_labels.tolist()
            phen_weights = model.final_proj.weight.data.tolist()[0]
            assert len(ids) == len(batch_text)
            assert len(ids) == len(labels_batch_list)
            assert len(ids) == len(output_list)
            assert len(ids) == len(attn_weights_list[0])
            for code in range(len(attn_weights_list)):
                for index in range(len(ids)):
                    temp_list = []
                    temp_list.append(ids[index])
                    temp_list.append(labels_batch_list[index][0])
                    temp_list.append(output_list[index][0])
                    temp_list.append(data_loader.icd_cols[code])
                    temp_list.append(icd_labels_list[index][code])
                    temp_list.append(phen_prob_list[index][code])
                    temp_list.append(phen_contr_list[index][code])
                    temp_list.append(phen_weights[code])
                    attn_words = list(zip(attn_weights_list[code][index], batch_text[index]))
                    attn_words.sort(reverse=True)
                    new_attn_words = [x for t in attn_words[:10] for x in t]
                    temp_list.extend(new_attn_words)
                    master_list.append(temp_list)

    df_attn_scores = pd.DataFrame(master_list,
                                  columns=["ICUSTAY_ID", params.task + "_label", params.task + "_prediction", 'Problem',
                                           'Problem_label', 'problem_prediction', 'problem_contribution',
                                           'phen_weight'] + [
                                              'attn_' + str(i // 2) if i % 2 == 0 else 'words_' + str(i // 2) for i in
                                              range(20)])
    print(df_attn_scores.dtypes)
    df_attn_scores.sort_values(by=[params.task + "_prediction", 'problem_contribution'], ascending=False, inplace=True)
    print(df_attn_scores.head(5))
    datasetPath = os.path.join(model_dir, 'df_high_attn_contribution.csv')
    df_attn_scores.to_csv(datasetPath, index=False)

    df_attn_scores.sort_values(by=[params.task + "_prediction", 'problem_prediction'], ascending=False, inplace=True)
    print(df_attn_scores.head(5))
    datasetPath = os.path.join(model_dir, 'df_high_attn_problem.csv')
    df_attn_scores.to_csv(datasetPath, index=False)

    logging.info("- Extracted attention : " + job_name)
    return


def allied_evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, allied=False):
    # set model to evaluation mode

    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    running_auc = utils.OutputAUC()
    running_metrics = utils.TestMetrics()
    running_icd = utils.MetricsICD()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for i in t:
            # fetch the next evaluation batch
            train_batch_w2v, train_batch_sp, labels_batch, icd_labels, ids = next(data_iterator)
            output_batch, icd_batch = model(train_batch_w2v)
            loss = loss_fn(output_batch, labels_batch)
            running_icd.update(icd_labels.data.cpu().numpy(), icd_batch.data.cpu().numpy())
            running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    metrics = running_metrics()
    logging.info('AUCROC: ' + str(running_auc()))
    logging.info('AUCPR: ' + str(metrics[1]))
    logging.info('AUCROC_ICD: ' + str(running_icd()))

    return {'AUCROC': metrics[0], "AUCPR": metrics[1], "AUCROC_ICD": running_icd()}


def allied_final_evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, allied=False):
    # set model to evaluation mode

    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    running_auc = utils.OutputAUC()
    running_metrics = utils.TestMetrics()
    running_icd = utils.MetricsICD()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for i in t:
            # fetch the next evaluation batch
            train_batch_w2v, train_batch_sp, labels_batch, icd_labels, ids = next(data_iterator)
            if 'w2v' in params.emb:
                output_batch, icd_batch = model(train_batch_w2v)
            elif 'sp' in params.emb:
                output_batch, icd_batch = model(train_batch_sp)
            else:
                output_batch, icd_batch = model(train_batch_w2v, train_batch_sp)
            loss = loss_fn(output_batch, labels_batch)
            # print(loss)
            running_icd.update(icd_labels.data.cpu().numpy(), icd_batch.data.cpu().numpy())
            running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data.item()
            summ.append(summary_batch)

    metrics = running_metrics()
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info('AUCROC' + str(running_auc()))
    logging.info('AUCROC' + str(metrics[0]))
    logging.info('AUCPR' + str(metrics[1]))
    logging.info('MICRO AUCROC_ICD' + str(running_icd()))
    macro_auc = running_icd.macro_auc()
    logging.info('MACRO AUCROC_ICD' + str(macro_auc))

    return {'AUCROC': metrics[0], "AUCPR": metrics[1], "MICRO_AUCROC_ICD": running_icd(), "MACRO_AUCROC_ICD": macro_auc}


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    # set model to evaluation mode

    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    running_auc = utils.OutputAUC()
    running_metrics = utils.TestMetrics()
    running_icd = utils.MetricsICD()
    # Use tqdm for progress bar
    with torch.no_grad():
        t = trange(num_steps)
        for i in t:
            # fetch the next evaluation batch
            if 'phen' in params.model:
                if params.task == 'icd':
                    train_batch_w2v, train_batch_sp, _, labels_batch = next(data_iterator)
                else:
                    train_batch_w2v, train_batch_sp, labels_batch, _ = next(data_iterator)
                output_batch = model(train_batch_w2v)
                loss = loss_fn(output_batch, labels_batch)
                loss = loss / params.grad_acc  # Normalize our loss (if averaged)
                # print(loss)
            elif params.model == 'lr':
                train_batch, labels_batch = next(data_iterator)
                output_batch = model(train_batch)
                loss = loss_fn(output_batch, labels_batch)

                loss = loss / params.grad_acc  # Normalize our loss (if averaged)
                # print(loss)
            else:
                train_batch_w2v, train_batch_sp, labels_batch, _, ids = next(data_iterator)
                output_batch = model(train_batch_w2v)
                loss = loss_fn(output_batch, labels_batch)

            running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            running_metrics.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            if params.task == 'icd_only':
                running_icd.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info('AUCROC' + str(running_auc()))

    if params.task == 'icd_only':
        return {'AUCROC': running_icd(), 'MACRO_AUCROC_ICD': running_icd.macro_auc()}
    else:
        logging.info('METRICS' + str(running_metrics()))
        metrics = running_metrics()
        return {'AUCROC': metrics[0], "AUCPR": metrics[1]}


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    master_results = []
    params.batch_size = 32
    params.grad_acc = 1
    results = {'AUCROC': [], 'AUCPR': [], 'MICRO_AUCROC_ICD': [], 'MACRO_AUCROC_ICD': []}
    for fold in range(0, 5):
        params.fold = fold

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        np.random.seed(230)
        np.random.seed(230)
        if params.cuda:
            torch.cuda.manual_seed_all(230)

        # Set the logger
        job_name = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_{}_{}".format(params.emb, params.learning_rate,
                                                               params.kernels,
                                                               params.batch_size, params.filters,
                                                               params.dropout,
                                                               params.model, params.task)
        if 'lstm' in params.model:
            job_name = "emb{}_lr{}_bs{}_h{}_dr{}_{}_{}".format(params.emb,
                                                               params.learning_rate,
                                                               params.batch_size,
                                                               params.h,
                                                               params.dropout,
                                                               params.model, params.task)
        elif params.model == 'lr':
            job_name = "lr{}_{}_icds{}>={}_{}".format(
                params.learning_rate,
                params.model,
                params.phenotype,
                params.icd_threshold,
                params.task)
        elif 'phen' in params.model:
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
            params.icd_model = "emb{}_lr{}_k{}_bs{}_f{}_dr{}_bn{}_{}_icds{}>={}_icdNum{}_{}".format(params.emb,
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
                                                                                                    'icd_only')
        # Create a new folder in parent_dir with unique_name "job_name"
        model_dir = os.path.join(params.save_path, 'fold' + str(params.fold), job_name)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        utils.set_logger(os.path.join(model_dir, 'eval1.log'))

        logging.info("Loading the datasets...")

        # load data
        data_loader = DataLoader(args.data_dir, params)

        data = data_loader.load_data(['train', 'test'], args.data_dir)
        train_data = data['train']
        test_data = data['test']
        params.num_phenotypes = train_data['num_phenotypes']

        # specify the train and val dataset sizes
        params.test_size = test_data['size']

        logging.info("- done.")

        # Define the model and optimizer
        device = torch.device("cuda:0" if params.cuda else sys.exit("gpu unavailable"))
        if params.model == "cnn_text":
            model = models.CNN_Text(data_loader.weights_w2v, params)
        elif params.model == "conv_attn_phen":
            model = models.CNN_Text_Attn_Phen(data_loader.weights_w2v, params)
        elif params.model == "lstm_attn":
            model = models.LSTM_Attn(data_loader.weights_w2v, params)
        elif params.model == "conv_attn":
            model = models.CNN_Text_Attn(data_loader.weights_w2v, params)
        elif params.model == 'lr':
            model = models.lr_baseline(params)

        print(model)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        # fetch loss function and metrics
        loss_fn = torch.nn.BCELoss()
        metrics = models.metrics

        # Evaluate the model
        logging.info("Starting evaluation ")
        utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model, parallel=False)
        num_steps = test_data['size'] // params.batch_size
        test_data_iterator = data_loader.data_iterator(test_data, params, shuffle=False)
        if 'icd_only' in params.task:
            test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
            results['MICRO_AUCROC_ICD'].append(test_metrics['AUCROC'])
            results['MACRO_AUCROC_ICD'].append(test_metrics['MACRO_AUCROC_ICD'])
        elif 'phec_attn' in params.eval_mode:
            test_metrics = evaluate_phec_attn(model, loss_fn, test_data_iterator, metrics, params,
                                              num_steps,
                                              data_loader, model_dir)
        elif 'allied' in params.eval_mode:
            test_metrics = allied_final_evaluate(model, loss_fn, test_data_iterator, metrics, params,
                                                 num_steps)
            print(model_dir)
            results['AUCROC'].append(test_metrics['AUCROC'])
            results['AUCPR'].append(test_metrics['AUCPR'])
            results['MICRO_AUCROC_ICD'].append(test_metrics['MICRO_AUCROC_ICD'])
            results['MACRO_AUCROC_ICD'].append(test_metrics['MACRO_AUCROC_ICD'])
        else:
            test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
            results['AUCROC'].append(test_metrics['AUCROC'])
            results['AUCPR'].append(test_metrics['AUCPR'])
            print(test_metrics['AUCROC'])
            print(test_metrics['AUCPR'])
            print(results)
    if 'icd_only' in params.task:
        master_results.append(
            ['{}_{}'.format(params.phen, params.task), np.mean(results['MICRO_AUCROC_ICD']),
             np.std(results['MICRO_AUCROC_ICD']), np.mean(results['MACRO_AUCROC_ICD']),
             np.std(results['MACRO_AUCROC_ICD'])])
        df_master_results = pd.DataFrame(master_results,
                                         columns=["model_name", 'MICRO_AUCROC_ICD_MEAN', 'MICRO_AUCROC_ICD_STD',
                                                  'MACRO_AUCROC_ICD_MEAN',
                                                  'MACRO_AUCROC_ICD_STD'])
        print(df_master_results.dtypes)
        print(df_master_results.head(5))
        datasetPath = os.path.join(params.save_path, 'df_master_final_results_icd_only.csv')
        df_master_results.to_csv(datasetPath, index=False)
    elif 'allied' in params.eval_mode:
        master_results.append(
            ['{}_{}'.format(params.model_name, params.task), np.mean(results['AUCROC']), np.std(results['AUCROC']),
             np.mean(results['AUCPR']), np.std(results['AUCPR']), np.mean(results['MICRO_AUCROC_ICD']),
             np.std(results['MICRO_AUCROC_ICD']), np.mean(results['MACRO_AUCROC_ICD']),
             np.std(results['MACRO_AUCROC_ICD'])])
        df_master_results = pd.DataFrame(master_results,
                                         columns=["model_name", 'AUCROC_MEAN', 'AUCROC_STD', 'AUCPR_MEAN', 'AUCPR_STD',
                                                  'MICRO_AUCROC_ICD_MEAN', 'MICRO_AUCROC_ICD_STD',
                                                  'MACRO_AUCROC_ICD_MEAN',
                                                  'MACRO_AUCROC_ICD_STD'])
        print(df_master_results.dtypes)
        print(df_master_results.head(5))
        datasetPath = os.path.join(params.save_path, 'df_master_final_results_{}.csv'.format(params.phenotype))
        df_master_results.to_csv(datasetPath, index=False)
    else:
        print(results['AUCROC'])
        print(results['AUCPR'])
        master_results.append(
            ['{}_{}'.format(params.model_name, params.task), np.mean(results['AUCROC']), np.std(results['AUCROC']),
             np.mean(results['AUCPR']), np.std(results['AUCPR'])])
        print(master_results)
        df_master_results = pd.DataFrame(master_results,
                                         columns=["model_name", 'AUCROC_MEAN', 'AUCROC_STD', 'AUCPR_MEAN', 'AUCPR_STD'])
        print(df_master_results.dtypes)
        print(df_master_results.head(5))
        datasetPath = os.path.join(params.save_path, 'df_master_final_results_{}.csv'.format(params.model))
        df_master_results.to_csv(datasetPath, index=False)
