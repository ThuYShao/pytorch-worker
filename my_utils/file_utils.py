# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import os

BASE_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Task2/processed'
DATA_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Case_Law/format'
WORK_DIR = '/data/disk2/private/guozhipeng/syq/coliee/pytorch-worker/'


def choose_start_point(_org_name, _out_name, _start):
    out_file = open(_out_name, 'w', encoding='utf-8')
    idx = 0
    with open(_org_name, encoding='utf-8') as fin:
        while True:
            json_str = fin.readline()
            if not json_str:
                break
            if idx < _start:
                continue
            out_file.write(json_str)
            idx += 1
    out_file.close()
    print('save in out_file=%s, #lines=%d' % (_out_name, idx))


if __name__ == '__main__':
    # merge file
    # out_file = open(os.path.join(BASE_DIR, 'data.json'), 'w', encoding='utf-8')
    # for folder in ['train', 'test']:
    #     in_file = os.path.join(BASE_DIR, folder + '.json')
    #     print(folder, '   start ...')
    #     with open(in_file, encoding='utf-8') as fin:
    #         while True:
    #             json_str = fin.readline()
    #             if not json_str:
    #                 break
    #             json_obj = json.loads(json_str.strip())
    #             json_obj['guid'] = json_obj['guid'] + '_%s' % folder
    #             out_line = json.dumps(json_obj, ensure_ascii=False) + '\n'
    #             out_file.write(out_line)
    # out_file.close()
    # print('save data all')
    org_name = os.path.join(DATA_DIR, 'train_body_pair_split.json')
    out_name = os.path.join(DATA_DIR, 'train_body_pair_left.json')
    choose_start_point(org_name, out_name, _start=100)

