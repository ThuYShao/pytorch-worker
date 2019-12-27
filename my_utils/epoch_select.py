# -*- coding: utf-8 -*-
__author__ = 'yshao'


import os
import json
import numpy as np

OUTPUT_DIR = 'output/model'


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
        print('type=%s, epoch=%d' % (type_, sort_lst[0][0]), sort_lst[0][1])


if __name__ == '__main__':
    model_path = 'attengru_bm25_tmp'
    file_name = os.path.join(OUTPUT_DIR, model_path, 'eval.json')
    eval_dict = load_json(file_name)

    select_best_epoch(eval_dict, 'f1')
