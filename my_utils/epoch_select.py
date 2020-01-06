# -*- coding: utf-8 -*-
__author__ = 'yshao'


import os
import json
import numpy as np

OUTPUT_DIR = '/data/disk2/private/guozhipeng/syq/coliee/pytorch-worker/output/model'


def load_json(_filename):
    json_str = open(_filename, encoding='utf-8').readline()
    json_obj = json.loads(json_str.strip())
    print('load eval dict from file = %s' % _filename)
    return json_obj


def select_best_epoch(_eval_dict, _key):
    print('select best epoch, key=%s' % _key)
    for type_ in _eval_dict:
        tmp_dict = _eval_dict[type_]
        sort_lst = sorted(tmp_dict.items(), key=lambda item: item[1][_key], reverse=True)
        print('type=%s, epoch=%d' % (type_, int(sort_lst[0][0])))
        for p_t in _eval_dict:
            print(p_t, _eval_dict[p_t][sort_lst[0][0]])


def early_stop(_eval_dict, _key, _th=10):
    print('early stop epoch, key=%s' % _key)
    if 'dev' not in _eval_dict:
        print('no dev set')
        return
    tmp_dict = _eval_dict['dev']
    sort_lst = sorted(tmp_dict.items(), key=lambda item: int(item[0]), reverse=False)
    if _key != 'loss':
        tmp_best = -1
        best_epoch = -1
        tmp_idx = -1
        for idx, item in enumerate(sort_lst):
            epoch_idx = int(item[0])
            score = item[1][_key]
            if score > tmp_best:
                tmp_best = score
                best_epoch = epoch_idx
                tmp_idx = idx
            if epoch_idx - best_epoch > _th:
                break
        print('epoch=%d, best_score=%f' % (best_epoch, tmp_best), sort_lst[tmp_idx])
        if 'test' in _eval_dict:
            print('on test set', _eval_dict['test']['%d' % best_epoch])
    else:
        tmp_best = 1000
        best_epoch = -1
        tmp_idx = -1
        for idx, item in enumerate(sort_lst):
            epoch_idx = int(item[0])
            score = item[1][_key]
            if score < tmp_best:
                tmp_best = score
                best_epoch = epoch_idx
                tmp_idx = idx
            if epoch_idx - best_epoch > _th:
                break
        print('epoch=%d, best_score=%f' % (best_epoch, tmp_best), sort_lst[tmp_idx])
        if 'test' in _eval_dict:
            print('on test set', _eval_dict['test']['%d' % best_epoch])


if __name__ == '__main__':
    model_path = 'main/attenlstm_bm25_1e4_decay_5'
    file_name = os.path.join(OUTPUT_DIR, model_path, 'eval.json')
    eval_dict = load_json(file_name)

    '''select epoch'''
    select_best_epoch(eval_dict, 'f1')
    # early_stop(eval_dict, 'f1', _th=10)
    # early_stop(eval_dict, 'loss', _th=10)
