import argparse
import pandas as pd
from pathlib import Path
import json
import numpy as np
from model.help_funcs import caption_evaluate
from transformers import BertTokenizer, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'
    
    line = ''
    if stds is None:
        for acc in accs:
            line += '{:0.1f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.1f}±{:0.1f} '.format(acc, std)
    
    if append_mean:
        line += '{:0.1f}'.format(sum(accs) / len(accs))
    print(category_line)
    print(line)


def get_mode(df):
    if 'bleu2' in df.columns:
        return 'caption'
    elif 'dataset0/bleu2' in df.columns:
        return 'mix_caption'
    elif 'test_inbatch_p2t_acc' in df.columns:
        return 'retrieval'
    elif 'onto_test_rerank_inbatch_p2t_rec20' in df.columns:
        return 'mix_retrieval'
    else:
        raise NotImplementedError


def read_retrieval(df, args):
    df = df.round(2)
    if args.disable_rerank:
        retrieval_cols = ['test_inbatch_p2t_acc', 'test_inbatch_p2t_rec20', 'test_inbatch_t2p_acc',   'test_inbatch_t2p_rec20',  'test_fullset_p2t_acc',  'test_fullset_p2t_rec20', 'test_fullset_t2p_acc', 'test_fullset_t2p_rec20']
    else:
        retrieval_cols = ['rerank_test_inbatch_p2t_acc', 'rerank_test_inbatch_p2t_rec20', 'rerank_test_inbatch_t2p_acc',   'rerank_test_inbatch_t2p_rec20', 'rerank_test_fullset_p2t_acc',  'rerank_test_fullset_p2t_rec20', 'rerank_test_fullset_t2p_acc', 'rerank_test_fullset_t2p_rec20']
    retrieval_log = df[~df['test_inbatch_t2p_acc'].isnull()][retrieval_cols]
    print(retrieval_cols)
    print(retrieval_log.to_string(header=False))

def read_mix_retrieval(df, args):
    df = df.round(2)

    ## dataset 1
    if args.disable_rerank:
        retrieval_cols = ['swiss_test_inbatch_p2t_acc', 'swiss_test_inbatch_p2t_rec20', 'swiss_test_inbatch_t2p_acc', 'swiss_test_inbatch_t2p_rec20','swiss_test_fullset_p2t_acc', 'swiss_test_fullset_p2t_rec20', 'swiss_test_fullset_t2p_acc', 'swiss_test_fullset_t2p_rec20']
    else:
        retrieval_cols = ['swiss_test_rerank_inbatch_p2t_acc', 'swiss_test_rerank_inbatch_p2t_rec20', 'swiss_test_rerank_inbatch_t2p_acc', 'swiss_test_rerank_inbatch_t2p_rec20', 'swiss_test_rerank_fullset_p2t_acc', 'swiss_test_rerank_fullset_p2t_rec20', 'swiss_test_rerank_fullset_t2p_acc', 'swiss_test_rerank_fullset_t2p_rec20']
    retrieval_log = df[~df['swiss_test_rerank_inbatch_p2t_acc'].isnull()][retrieval_cols]
    print(retrieval_cols)
    print(retrieval_log.to_string(header=False))
    print('--------------------')

    if args.disable_rerank:
        retrieval_cols = ['onto_test_inbatch_p2t_acc', 'onto_test_inbatch_p2t_rec20', 'onto_test_inbatch_t2p_acc', 'onto_test_inbatch_t2p_rec20',
        'onto_test_fullset_p2t_acc', 'onto_test_fullset_p2t_rec20', 'onto_test_fullset_t2p_acc', 'onto_test_fullset_t2p_rec20']
    else:
        retrieval_cols = ['onto_test_rerank_inbatch_p2t_acc','onto_test_rerank_inbatch_p2t_rec20','onto_test_rerank_inbatch_t2p_acc','onto_test_rerank_inbatch_t2p_rec20', 'onto_test_rerank_fullset_p2t_acc','onto_test_rerank_fullset_p2t_rec20','onto_test_rerank_fullset_t2p_acc','onto_test_rerank_fullset_t2p_rec20']
    
    retrieval_log = df[~df['onto_test_rerank_inbatch_p2t_acc'].isnull()][retrieval_cols]
    print(retrieval_cols)
    print(retrieval_log.to_string(header=False))
    
    
    



def read_caption(df, args):
    df = df.round(2)
    df = df[~df['bleu2'].isnull()]
    if 'acc' in df.columns:
        cols = ['epoch', 'acc', 'bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
    else:
        cols = ['epoch', 'bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
    caption_log = df[cols]
    
    print(cols)
    print(caption_log)


def read_mix_caption(df, args):
    df = df.round(2)
    df = df[~df['dataset0/bleu2'].isnull()]
    cols = ['epoch', 'dataset0/acc', 'dataset0/bleu2','dataset0/bleu4','dataset0/rouge_1','dataset0/rouge_2','dataset0/rouge_l','dataset0/meteor_score']
    caption_log = df[cols]
    print('dataset 0')
    print([col.split('/')[-1] for col in cols])
    print(caption_log.to_string(header=False))
    if 'dataset1/acc' in df.columns:
        print('------------------------------')
        cols = ['epoch', 'dataset1/acc', 'dataset1/bleu2','dataset1/bleu4','dataset1/rouge_1','dataset1/rouge_2','dataset1/rouge_l','dataset1/meteor_score']

        caption_log = df[cols]
        print('dataset 1')
        print([col.split('/')[-1] for col in cols])
        print(caption_log.to_string(header=False))
    

def exact_match(prediction_list, target_list):
    match = 0
    for prediction, target in zip(prediction_list, target_list):
        prediction = prediction.strip()
        target = target.strip()
        if prediction == target:
            match += 1
    acc = round(match / len(prediction_list) * 100, 2)
    return acc

def read_caption_prediction(args):
    path = args.path
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    # tokenizer = BertTokenizer.from_pretrained('facebook/galactica-1.3b')
    # tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-1.3b')
    tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-1.3b', use_fast=False, padding_side='right')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    prediction_list = []
    target_list = []
    for line in lines:
        prediction = line['prediction'].strip()
        target = line['target'].strip()
        prediction_list.append(prediction)
        target_list.append(target)
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = caption_evaluate(prediction_list, target_list, tokenizer, 128)
    bleu2 = round(bleu2, 2)
    bleu4 = round(bleu4, 2)
    rouge_1 = round(rouge_1, 2)
    rouge_2 = round(rouge_2, 2)
    rouge_l = round(rouge_l, 2)
    meteor_score = round(meteor_score, 2)
    
    acc = exact_match(prediction_list, target_list)
    cols = ['Exact match', 'bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
    print(cols)
    print(acc, bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score)


def read_mpp_results(args):
    ds_list = ['bace', 'bbbp', 'clintox', 'toxcast', 'sider', 'tox21']
    from pathlib import Path
    results = []
    stds = []
    used_ds = []
    for ds in ds_list:
        ds_path = Path(args.path) / ds
        if not ds_path.exists():
            continue
        ds_path = ds_path / 'lightning_logs'
        test_roc_list = []
        for f in ds_path.glob("version_*"):
            f = f / 'metrics.csv'
            df = pd.read_csv(f)
            df = df[['val roc', 'test roc']]
            df = df[~df['val roc'].isnull()]
            array = df.to_numpy()
            test_roc = array[array[:, 0].argmax(), 1]
            test_roc_list.append(test_roc)
        test_roc_list = np.asarray(test_roc_list)
        test_roc = round(test_roc_list.mean() * 100, 2)
        results.append(test_roc)
        test_std = round(test_roc_list.std() * 100, 2)
        stds.append(test_std)
        used_ds.append(ds)

    print_std(results, stds, used_ds, True)

def read_regression_results(args):
    path = Path(args.path)
    test_rmse_list = []
    for file in path.glob('version_*'):
        file = file / 'metrics.csv'
        df = pd.read_csv(file)
        df = df[['val rmse', 'test rmse']]
        df = df[~df['val rmse'].isnull()]
        array = df.to_numpy()
        test_rmse = array[array[:, 0].argmin(), 1]
        test_rmse_list.append(test_rmse)
    test_rmse_list = np.asarray(test_rmse_list)
    mean = round(test_rmse_list.mean(), 3)
    std = round(test_rmse_list.std(), 3)
    print(f'{mean}±{std}')


def read_qa_results(path, text_trunc_length):
    tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-1.3b', use_fast=False, padding_side='right')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [json.loads(line) for line in lines]
    for line in lines:
        line['target'] = line['target'].strip()
        line['prediction'] = line['prediction'].strip()
        
    ## overall accuracy first
    total = len(lines)
    correct = 0
    for line in lines:
        if line['target'] == line['prediction']:
            correct += 1
    overall_acc = round(correct / total * 100, 2)
    
    ## accuracy for each type
    q_types = ['Number structure/property', 'Number side information', 'String structure/property', 'String side information', 'Number', 'String']
    q_type2acc = {q_type: 0 for q_type in q_types}
    q_type2bleu2 = {q_type: 0 for q_type in q_types if q_type.find('String') >= 0}
    for q_type in q_types:
        prediction_list = []
        target_list = []
        for line in lines:
            if line['q_type'].find(q_type) >= 0:
                prediction_list.append(line['prediction'])
                target_list.append(line['target'])
        if len(prediction_list) == 0:
            continue
        correct = 0
        for prediction, target in zip(prediction_list, target_list):
            if prediction == target:
                correct += 1
        acc = round(correct / len(prediction_list) * 100, 2)
        q_type2acc[q_type] = acc
        
        ## get bleu-2 score for string questions
        if q_type.find('String') >= 0:
            # print(prediction_list)
            prediction_list = tokenizer(prediction_list, truncation=True, max_length=text_trunc_length, padding=False)['input_ids']
            prediction_list = [tokenizer.convert_ids_to_tokens(i) for i in prediction_list]
            target_list = tokenizer(target_list, truncation=True, max_length=text_trunc_length, padding=False)['input_ids']
            target_list = [tokenizer.convert_ids_to_tokens(i) for i in target_list]
            
            target_list = list(filter(('<pad>').__ne__, target_list))
            target_list = list(filter(('[PAD]').__ne__, target_list))
            target_list = list(filter(('[CLS]').__ne__, target_list))
            target_list = list(filter(('[SEP]').__ne__, target_list))
            
            prediction_list = list(filter(('<pad>').__ne__, prediction_list))
            prediction_list = list(filter(('[PAD]').__ne__, prediction_list))
            prediction_list = list(filter(('[CLS]').__ne__, prediction_list))
            prediction_list = list(filter(('[SEP]').__ne__, prediction_list))

            hypothesis = prediction_list
            references = [[t] for t in target_list]
            bleu2 = corpus_bleu(references, hypothesis, weights=(.5,.5))
            bleu2 = round(bleu2 * 100, 2)
            q_type2bleu2[q_type] = bleu2
    print('overall accuracy')
    print(overall_acc, )
    print('accuracy')
    print(q_type2acc)
    print('bleu-2')
    print(q_type2bleu2)
    return overall_acc, q_type2acc, q_type2bleu2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--disable_rerank', action='store_true', default=False)
    parser.add_argument('--qa_question', action='store_true', default=False)
    args = parser.parse_args()
    args.path = Path(args.path)
    
    if args.qa_question:
        read_qa_results(args.path, 128)
        exit()
    
    if args.path.name.find('predictions') >= 0:
        read_caption_prediction(args)
        exit()
    elif str(args.path).find('mpp') >= 0:
        read_mpp_results(args)
        exit()
    elif str(args.path).find('regression') >= 0:
        read_regression_results(args)
        exit()

    log_hparas = args.path / 'hparams.yaml'
    with open(log_hparas, 'r') as f:
        line = f.readline()
        file_name = line.strip().split(' ')[1]
    
    log_path = args.path / 'metrics.csv'
    log = pd.read_csv(log_path)
    
    print(f'File name: {file_name}')
    mode = get_mode(log)
    
    if mode == 'retrieval':
        read_retrieval(log, args)
    elif mode == 'caption':
        read_caption(log, args)
    elif mode == 'mix_caption':
        read_mix_caption(log, args)
    elif mode == 'mix_retrieval':
        read_mix_retrieval(log, args)