import os
import torch
from load_data import data_loader
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import logging

class Tester(object):
    def __init__(self, opt):

        self.opt = opt

        self.loader = data_loader(opt)
        self.generator = data.DataLoader(self.loader, batch_size=opt['batch_size'], shuffle=True, drop_last=False)

        # model
        self.G = torch.load(self.opt['save_path'] + 'G_' + str(self.opt['epoch']) + '.pth')
        self.D = torch.load(self.opt['save_path'] + 'D_' + str(self.opt['epoch']) + '.pth')

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        if opt['cuda']:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.G_loss = self.G_loss.cuda()
            self.D_loss = self.D_loss.cuda()

        # Optimizer
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=opt['lr'])
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=opt['lr'])

    def test(self):

        self.G.eval()
        self.D.eval()
        result = torch.zeros((self.loader.time_num, self.loader.node_num, 4))
        for step, ((recent_data, trend_data, time_feature), sub_graph, real_returns, index_t, index_r) in enumerate(
                self.generator):
            """
            recent_data: (batch_size, time, node_num, num_feature)
            trend_data: (batch_size, time, num_feature)
            real_returns: (batch_size, num_adj, num_feature)
            """
            if self.opt['cuda']:
                recent_data, trend_data, real_returns, sub_graph, time_feature = \
                    recent_data.cuda(), trend_data.cuda(), real_returns.cuda(), sub_graph.cuda(), time_feature.cuda()

            real_sequence = torch.cat([recent_data, real_returns.unsqueeze(1)], dim=1)
            predicted_returns = self.G(recent_data, trend_data, sub_graph, time_feature)

            fake_sequence = torch.cat([recent_data, predicted_returns.unsqueeze(1)], dim=1)
            mse_loss = torch.pow(predicted_returns - real_returns, 2)
            
            # Direction accuracy: did we predict the right sign?
            direction_correct = torch.sign(predicted_returns) == torch.sign(real_returns)

            real_score_D = self.D(real_sequence, sub_graph, trend_data)
            fake_score_D = self.D(fake_sequence, sub_graph, trend_data)

            batch_size = recent_data.shape[0]
            for b in range(batch_size):
                result[index_t[b].item(), index_r[b].item(), 0] = torch.mean(mse_loss[b, ]).item()
                result[index_t[b].item(), index_r[b].item(), 1] = torch.mean(direction_correct[b, ].float()).item()
                result[index_t[b].item(), index_r[b].item(), 2] = real_score_D[b].item()
                result[index_t[b].item(), index_r[b].item(), 3] = fake_score_D[b].item()

        directory = self.opt['result_path']
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        np.save(directory + 'result' + '.npy', result.cpu().numpy())