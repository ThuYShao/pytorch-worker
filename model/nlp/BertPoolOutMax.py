# -*- coding: utf-8 -*-
__author__ = 'yshao'


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class BertPoolOutMax(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPoolOutMax, self).__init__()
        self.max_para_c = config.getint('model', 'max_para_c')
        self.max_para_q = config.getint('model', 'max_para_q')
        self.max_len = config.getint("data", "max_seq_length")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_c))

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        # batch_size = input_ids.size()[0]
        with torch.no_grad():
            input_ids = input_ids.view(-1, self.max_len)
            attention_mask = attention_mask.view(-1, self.max_len)
            token_type_ids = token_type_ids.view(-1, self.max_len)

            print('before bert', input_ids.size(), attention_mask.size(), token_type_ids.size())

            input('continue?')

            _, y = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                             output_all_encoded_layers=False)

            y = y.view(-1, self.max_para_q, self.max_para_c, 768)

            y = y.permute(0, 3, 1, 2)

            y = self.maxpool(y)

            y = y.view(-1, 768, self.max_para_q, 1)

            y = y.squeeze(3)

            y = y.permute(0, 2, 1)

            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}
        