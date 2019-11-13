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
        self.maxpool = nn.MaxPool1d(kernel_size=self.max_para_c)

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        # print(input_ids.size(), attention_mask.size(), token_type_ids.size())
        # batch_size = input_ids.size()[0]
        with torch.no_grad():
            # input_ids = input_ids.view(-1, self.max_len)
            # attention_mask = attention_mask.view(-1, self.max_len)
            # token_type_ids = token_type_ids.view(-1, self.max_len)

            for k in range(input_ids.size()[0]):
                q_lst = []
                for i in range(self.max_para_q):
                    input_ids_row = input_ids[k, i].view(-1, self.max_len)
                    attention_mask_row = attention_mask[k, i].view(-1, self.max_len)
                    token_type_ids_row = token_type_ids[k, i].view(-1, self.max_len)
                    print(input_ids_row.size(), attention_mask_row.size(), token_type_ids_row.size())
                    _, lst = self.bert(input_ids_row, token_type_ids=token_type_ids_row,
                                       attention_mask=attention_mask_row)
                    # lst = []
                    # for j in range(self.max_para_c):
                    #     _, y = self.bert(input_ids[k, i, j].unsqueeze(0), token_type_ids=token_type_ids[k, i, j].unsqueeze(0),
                    #                      attention_mask=attention_mask[k, i, j].unsqueeze(0))
                    #     y = y.view(1, -1)
                    #     lst.append(y)
                    # lst = torch.cat(lst, dim=0)
                    # print('after concat', lst.size())
                    lst = lst.view(self.max_para_c, -1)
                    lst = lst.transpose(0, 1)
                    print('after transpose', lst.size())
                    lst = lst.unsqueeze(0)
                    print('after unsqueeze', lst.size())
                    max_out = self.maxpool(lst)
                    max_out = max_out.squeeze()
                    print('max out size', max_out.size())
                    q_lst.append(max_out.detach().cpu().tolist())
                    input('continue?')
                print(len(q_lst))
            # input('continue to print content of q_list?')
            # print(q_lst)


            # _, y = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            #                  output_all_encoded_layers=False)
            #
            # y = y.view(-1, self.max_para_q, self.max_para_c, 768)
            #
            # y = y.permute(0, 3, 1, 2)
            #
            # y = self.maxpool(y)
            #
            # y = y.view(-1, 768, self.max_para_q, 1)
            #
            # y = y.squeeze(3)
            #
            # y = y.permute(0, 2, 1)

            # output = []
            # y = y.cpu().detach().numpy().tolist()
            # for i, guid in enumerate(data['guid']):
            #     output.append([guid, y[i]])
            # return {"output": output}
        
