# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import os

BASE_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Task2/processed'

if __name__ == '__main__':
    # merge file
    out_file = open(os.path.join(BASE_DIR, 'data.json'), 'w', encoding='utf-8')
    for folder in ['train', 'test']:
        in_file = os.path.join(BASE_DIR, folder + '.json')
        print(folder, '   start ...')
        with open(in_file, encoding='utf-8') as fin:
            while True:
                json_str = fin.readline()
                if not json_str:
                    break
                json_obj = json.loads(json_str.strip())
                json_obj['guid'] = json_obj['guid'] + '_%s' % folder
                out_line = json.dumps(json_obj, ensure_ascii=False) + '\n'
                out_file.write(out_line)
    out_file.close()
    print('save data all')
