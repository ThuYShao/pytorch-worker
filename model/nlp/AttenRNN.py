# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        # hidden: B * M * H, feature: B * H * 1
        ratio = torch.bmm(hidden, feature)
        # ratio: B * M * 1
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).unsqueeze(2)
        # result: B * H
        result = torch.bmm(hidden.permute(0, 2, 1), ratio)
        result = result.view(result.size(0), -1)
        return result


class AttentionRNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(AttentionRNN, self).__init__()

        self.input_dim = 768
        self.hidden_dim = config.getint('model', 'hidden_dim')
        self.output_dim = config.getint("model", "output_dim")
        self.max_para_q = config.getint('model', 'max_para_q')

        if config.get('model', 'rnn') == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=1, bidirectional=True)
        else:
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, num_layers=1, bidirectional=True)

        self.max_pool = nn.MaxPool1d(kernel_size=self.max_para_q)
        self.fc_a = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.attention = Attention(config)
        self.fc_f = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_hidden(self, config, gpu_list):
        if torch.cuda.is_available() and len(gpu_list) > 0:
            if config.get('model', 'rnn') == 'lstm':
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                self.hidden_dim).cuda()),
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                    self.hidden_dim).cuda())
                )
            else:
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                    self.hidden_dim).cuda())
                )
        else:
            if config.get('model', 'rnn') == 'lstm':
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                self.hidden_dim)),
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                    self.hidden_dim))
                )
            else:
                self.hidden = (
                    torch.autograd.Variable(
                        torch.zeros(2, config.getint("train", "batch_size"),
                                    self.hidden_dim))
                )

    def init_multi_gpu(self, device, config, *args, **params):
        self.rnn = nn.DataParallel(self.rnn, device_ids=device)
        self.max_pool = nn.DataParallel(self.max_pool, device_ids=device)
        self.fc_a = nn.DataParallel(self.fc_a, device_ids=device)
        self.attention = nn.DataParallel(self.attention, device_ids=device)
        self.fc_f = nn.DataParallel(self.fc_f, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input'] # B * M * I
        self.init_hidden(config, gpu_list) # 2 * B * H
        rnn_out, self.hidden = self.rnn(x, self.hidden) # rnn_out: B * M * 2H, hidden: 2 * B * H

        tmp_rnn = rnn_out.permute(0, 2, 1) # B * 2H * M

        feature = self.max_pool(tmp_rnn) # B * 2H * 1
        feature = feature.squeeze(2) # B * 2H
        feature = self.fc_a(feature) # B * 2H
        feature = feature.unsqueeze(2) # B * 2H * 1

        atten_out = self.attention(feature, rnn_out) # B * (2H)

        y = self.fc_f(atten_out)
        y = y.view(y.size()[0], -1)

        if 'label' in data.keys():
            label = data['label']
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}
        else:
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}










