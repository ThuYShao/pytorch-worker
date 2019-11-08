# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import os
import numpy as np
from collections import defaultdict

BASE_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Task2'


def load_result(_filename):
    res_dict = defaultdict(lambda: 0)
    res_obj = json.load(open(_filename, encoding='utf-8'))
    for item in res_obj:
        guid = item[0]
        res_lst = item[1]
        label = np.argmax(res_lst)
        res_dict[guid] = label
    print('load all results from file=%s, #res=%d' % (_filename, len(res_dict)))
    return res_dict


def load_label(_filename):
    label_dict = json.load(open(_filename, encoding='utf-8'))
    print('load labels from file=%s, #query=%d' % (_filename, len(label_dict)))
    return label_dict


def evaluate_label(_res_dict, _label_dict):
    correct = 0
    label = 0
    predict = 0
    for qid in _label_dict:
        label += len(_label_dict[qid])
    for guid in _label_dict:
        qid, cid = guid.split('_')
        if qid in _label_dict and cid in _label_dict[qid]:
            correct += 1
        predict += 1
    print('#correct=%d, #label=%d, #predict=%d' % (correct, label, predict))

    precision = float(correct) / predict
    recall = float(correct) / label
    if precision > 0 or recall > 0:
        F1 = (2 * precision * recall) / (precision + recall)
    else:
        F1 = 0
    print('precision=%f, recall=%f, f1=%f' % (precision, recall, F1))
    return precision, recall, F1


if __name__ == '__main__':
    label_file = os.path.join(BASE_DIR, 'test_labels.json')
    res_file = './result/task2_point_test.json'
    label_dict = load_label(label_file)
    res_dict = load_result(res_file)
    evaluate_label(res_dict, label_dict)



