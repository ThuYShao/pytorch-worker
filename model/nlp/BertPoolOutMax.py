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
        # self.maxpool = nn.MaxPool1d(kernel_size=self.max_para_c)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_c))

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']

        with torch.no_grad():
            output = []
            for k in range(input_ids.size()[0]):
                q_lst = []
                for i in range(self.max_para_q, step=2):
                    _, lst = self.bert(input_ids[k, i:i+1].view(-1, self.max_len),
                                       token_type_ids=token_type_ids[k, i:i+1].view(-1, self.max_len),
                                       attention_mask=attention_mask[k, i:i+1].view(-1, self.max_len))
                    print('before view', lst.size())
                    lst = lst.view(2, self.max_para_c, -1)
                    print('after view', lst.size())
                    lst = lst.permute(2, 0, 1)
                    print('after permute', lst.size)
                    # lst = lst.transpose(0, 1)
                    lst = lst.unsqueeze(0)
                    print('after unsquezze', lst.size())
                    max_out = self.maxpool(lst)
                    print('after maxpool', lst.size())
                    max_out = max_out.squeeze()
                    print('after squeeze', lst.size())
                    max_out = max_out.transpose(0, 1)
                    q_lst.extend(max_out.detach().cpu().tolist())
                print(len(q_lst))
                exit()
                assert (len(q_lst) == self.max_para_q)
                output.append([data['guid'][k], q_lst])
            return {"output": output}
        
