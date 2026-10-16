import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import torch
from random import random
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test', eopch=-1, mode=None):
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    total_int_pred = []
    total_int_label = []
    for i, data in enumerate(val_iter):
        model.set_input(data)
        model.test()
        emo_pred = model.emo_pred.argmax(dim=1).detach().cpu().numpy()
        int_pred = model.int_pred.argmax(dim=1).detach().cpu().numpy()
        emo_label = data['emo_label']
        int_label = data['int_label']
        total_emo_pred.append(emo_pred)
        total_emo_label.append(emo_label)
        total_int_pred.append(int_pred)
        total_int_label.append(int_label)
    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)
    total_int_pred = np.concatenate(total_int_pred)
    total_int_label = np.concatenate(total_int_label)
    emo_acc = accuracy_score(total_emo_label, total_emo_pred)
    int_acc = accuracy_score(total_int_label, total_int_pred)
    emo_uar = recall_score(total_emo_label, total_emo_pred, average='macro')
    int_uar = recall_score(total_int_label, total_int_pred, average='macro')
    emo_waf = f1_score(total_emo_label, total_emo_pred, average='weighted')
    int_waf = f1_score(total_int_label, total_int_pred, average='weighted')
    emo_micro_f1 = f1_score(total_emo_label, total_emo_pred, average='micro')
    int_micro_f1 = f1_score(total_int_label, total_int_pred, average='micro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)
    int_cm = confusion_matrix(total_int_label, total_int_pred)
    model.train()
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_emo_pred.npy'.format(phase)), total_emo_pred)
        np.save(os.path.join(save_dir, '{}_emo_label.npy'.format(phase)), total_emo_label)
        np.save(os.path.join(save_dir, '{}_int_pred.npy'.format(phase)), total_int_pred)
        np.save(os.path.join(save_dir, '{}_int_label.npy'.format(phase)), total_int_label)
    return (emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1, emo_cm, int_cm)

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))
if __name__ == '__main__':
    opt = Options().parse()
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    total_cv = 3 if opt.corpus_name != 'MSP' else 12
    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'), total_cv=total_cv)
    suffix = '_'.join([opt.model, opt.dataset_mode])
    logger = get_logger(logger_path, suffix)
    if opt.has_test:
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['train', 'valid', 'test'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'valid'])
    dataset_size = len(dataset)
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    model.logger = logger
    total_iters = 0
    best_emo_acc, best_emo_uar, best_int_acc, best_int_uar = (0, 0, 0, 0)
    best_emo_waf, best_int_waf, best_emo_micro_f1, best_int_micro_f1 = (0, 0, 0, 0)
    best_eval_epoch = -1
    best_loss = 100
    shared_miss_point = []
    shared_point = []
    shared_num = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        total_loss = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch)
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)
        emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1, emo_cm, int_cm = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d emo_acc %.4f int_acc %.4f emo_uar %.4f int_uar %.4f emo_waf %.4f int_waf %.4f emo_micro_f1 %.4f int_micro_f1 %.4f' % (epoch, opt.niter + opt.niter_decay, emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1))
        logger.info('\n{}'.format(emo_cm))
        logger.info('\n{}'.format(int_cm))
        if opt.has_test and opt.verbose:
            emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1, emo_cm, int_cm = eval(model, tst_dataset)
            logger.info('Tst result of epoch %d / %d emo_acc %.4f int_acc %.4f emo_uar %.4f int_uar %.4f emo_waf %.4f int_waf %.4f emo_micro_f1 %.4f int_micro_f1 %.4f' % (epoch, opt.niter + opt.niter_decay, emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1))
            logger.info('\n{}'.format(emo_cm))
            logger.info('\n{}'.format(int_cm))
        if emo_uar * int_uar > best_emo_uar * best_int_uar:
            best_eval_epoch = epoch
            best_emo_uar = emo_uar
            best_emo_acc = emo_acc
            best_int_uar = int_uar
            best_int_acc = int_acc
            best_emo_waf = emo_waf
            best_int_waf = int_waf
            best_emo_micro_f1 = emo_micro_f1
            best_int_micro_f1 = int_micro_f1
        select_metric = 'uar'
        best_metric = best_emo_uar + best_int_uar
    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1, emo_cm, int_cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result emo_acc %.4f int_acc %.4f emo_uar %.4f int_uar %.4f emo_waf %.4f int_waf %.4f emo_micro_f1 %.4f int_micro_f1 %.4f' % (emo_acc, int_acc, emo_uar, int_uar, emo_waf, int_waf, emo_micro_f1, int_micro_f1))
        logger.info('\n{}'.format(emo_cm))
        logger.info('\n{}'.format(int_cm))
        result_recorder.write_result_to_tsv({'emo_acc': emo_acc, 'int_acc': int_acc, 'emo_uar': emo_uar, 'int_uar': int_uar, 'emo_waf': emo_waf, 'int_waf': int_waf, 'emo_micro_f1': emo_micro_f1, 'int_micro_f1': int_micro_f1}, cvNo=opt.cvNo)
    else:
        result_recorder.write_result_to_tsv({'emo_acc': emo_acc, 'int_acc': best_int_acc, 'emo_uar': best_emo_uar, 'int_uar': best_int_uar, 'emo_waf': best_emo_waf, 'int_waf': best_int_waf, 'emo_micro_f1': best_emo_micro_f1, 'int_micro_f1': best_int_micro_f1}, cvNo=opt.cvNo)
    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
