"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import model.models as models
from model.data_loader import DataLoader
from evaluate import evaluate, allied_evaluate
import torch.nn as nn
import random
import sys
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    running_auc = utils.OutputAUC()
    running_auc_icds = utils.MetricsICD()
    loss = 0
    for i in t:
        # fetch the next training batch
        if 'phen' in params.model:
            train_batch_w2v, train_batch_sp, labels_batch, _ = next(data_iterator)
            output_batch = model(train_batch_w2v)
            loss = loss_fn(output_batch, labels_batch)
            loss = loss / params.grad_acc  # Normalize our loss (if averaged)

        elif params.model == 'lr':
            train_batch, labels_batch = next(data_iterator)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss = loss / params.grad_acc  # Normalize our loss (if averaged)
        else:
            train_batch_w2v, train_batch_sp, labels_batch, _, ids = next(data_iterator)
            output_batch = model(train_batch_w2v)
            loss = loss_fn(output_batch, labels_batch)
            loss = loss / params.grad_acc  # Normalize our loss (if averaged)

        running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())

        # compute gradients of all variables wrt loss
        loss.backward()

        if i % params.grad_acc == 0:
            # performs updates using calculated gradients
            optimizer.step()
            # clear previous gradients
            optimizer.zero_grad()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}

            summary_batch['loss'] = float(loss.data.item())
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(float(loss.data.item()))
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    logging.info('Train AUC: ' + str(running_auc()))
    return loss_avg()


def allied_train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps, train_target=False):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    running_auc = utils.OutputAUC()
    running_auc_icds = utils.MetricsICD()
    for i in t:
        # fetch the next training batch
        train_batch_w2v, train_batch_sp, labels_batch, icd_labels, ids = next(data_iterator)

        output_batch, icd_batch = model(train_batch_w2v)
        if train_target:
            loss = loss_fn(output_batch, labels_batch) + loss_fn(icd_batch, icd_labels)
        else:
            loss = loss_fn(icd_batch, icd_labels)
        loss = loss / params.grad_acc  # Normalize our loss (if averaged)

        running_auc.update(labels_batch.data.cpu().numpy(), output_batch.data.cpu().numpy())
        running_auc_icds.update(icd_labels.data.cpu().numpy(), icd_batch.data.cpu().numpy())

        # compute gradients of all variables wrt loss
        loss.backward()

        if i % params.grad_acc == 0:
            # performs updates using calculated gradients
            optimizer.step()
            # clear previous gradients
            optimizer.zero_grad()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}

            summary_batch['loss'] = float(loss.data.item())
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(float(loss.data.item()))
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    logging.info('Train AUC: ' + str(running_auc()))
    logging.info('Train ICD AUC: ' + str(running_auc_icds()))
    return loss_avg()


def train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, metrics, params, model_dir,
                       data_loader, restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    if 'phen' in params.model and 'frozen' in params.task:
        restore_path = os.path.join(params.save_path, 'fold' + str(params.fold), params.icd_model, 'best.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model)
        model.freeze()
    best_val_auc = 0
    best_icd_auc = 0
    num_epochs = 0
    epoch = 0

    train_target = False
    while num_epochs < params.patience and epoch < params.num_epochs:
        # Run one epoch
        logging.info(model_dir)
        logging.info("Epoch {}".format(epoch + 1))
        logging.info("Patience: {}".format(num_epochs + 1))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = train_data['size'] // params.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
        if 'allied' in params.task:
            logging.info("Training Target Outcome" if train_target else 'Training Only Phenotype Prediction')
            allied_train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps,
                         train_target)
            # Evaluate for one epoch on validation set
            num_steps = val_data['size'] // params.batch_size
            val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
            val_metrics = allied_evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
            best_icd_auc = max(best_icd_auc, val_metrics['AUCROC_ICD'])
            logging.info("Current ICD AUC: " + str(val_metrics['AUCROC_ICD']))
            logging.info("Best ICD AUC: " + str(best_icd_auc))

            if val_metrics['AUCROC_ICD'] >= params.icd_val_threshold:
                train_target = True
            else:
                train_target = False
                num_epochs = -1
        else:
            train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)
            # Evaluate for one epoch on validation set
            num_steps = val_data['size'] // params.batch_size
            val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
            val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)

        is_best = np.mean(val_metrics['AUCROC']) >= best_val_auc
        model_to_save = model
        state_dict = model_to_save.state_dict()
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': state_dict,
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
        if params.task == 'icd_only':
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': state_dict,
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=os.path.join(params.save_path, 'fold' + str(params.fold),
                                                          params.icd_model))

        if is_best:
            logging.info("- Found new best auc")
            best_val_auc = np.mean(val_metrics['AUCROC'])

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            num_epochs = -1

        num_epochs += 1
        epoch += 1

        # Save latest val metrics in a json file in the model directory
        logging.info("Current AUC: " + str(val_metrics['AUCROC']))
        logging.info("Best AUC: " + str(best_val_auc))
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    if params.test_mode:
        utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model, parallel=True)
        num_steps = test_data['size'] // params.batch_size
        test_data_iterator = data_loader.data_iterator(test_data, params, shuffle=False)
        if 'allied' in params.task:
            test_metrics = allied_evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
        else:
            test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
    logging.info("TEST METRICS: " + str(test_metrics['AUCROC']))
    logging.info("MEAN TEST METRICS: " + str(np.mean(test_metrics['AUCROC'])))


if __name__ == '__main__':
    torch.set_num_threads(4)
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if params.cuda:
        torch.cuda.manual_seed_all(230)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if params.cuda:
        torch.cuda.manual_seed_all(230)

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
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    logging.info(job_name)
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)

    data = data_loader.load_data(['train', 'val', 'test'], args.data_dir)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    params.num_phenotypes = train_data['num_phenotypes']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Define the model and optimizer
    device = torch.device("cuda:0" if params.cuda else sys.exit("gpu unavailable"))
    if params.model == "cnn_text":
        model = models.CNN_Text(data_loader.weights_w2v, params)
    elif params.model == "lstm_attn":
        model = models.LSTM_Attn(data_loader.weights_w2v, params)
    elif params.model == "conv_attn":
        model = models.CNN_Text_Attn(data_loader.weights_w2v, params)
    elif params.model == "conv_attn_phen":
        model = models.CNN_Text_Attn_Phen(data_loader.weights_w2v, params)
    elif params.model == 'lr':
        model = models.lr_baseline(params)

    print(model)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # fetch loss function and metrics
    loss_fn = torch.nn.BCELoss()
    metrics = models.metrics

    # Train the model
    logging.info("Starting training ")
    train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, metrics, params,
                       model_dir, data_loader,
                       args.restore_file)

    print(model_dir)
