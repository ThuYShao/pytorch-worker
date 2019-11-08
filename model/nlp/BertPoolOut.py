# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class BertPoolOut(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPoolOut, self).__init__()
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        with torch.no_grad():
            _, y = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                             output_all_encoded_layers=False)
            y = y.view(y.size()[0], -1)


        output = {
            'guid': data['guid'],
            'pooled_out': y.detach().cpu().numpy().tolist()
        }

        return {'output': output}
