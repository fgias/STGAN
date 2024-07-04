import argparse
import torch
import numpy as np
import random
import os
import datetime
from trainer import Trainer
from tester import Tester
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger()

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bay', help='bay or nyc')
parser.add_argument('--root_path', type=str, default='./', help='root path: dataset, checkpoint')

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
parser.add_argument('--epoch', type=int, default=6, help='Number of training epochs per iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lambda_G', type=int, default=500, help='lambda_G for generator loss function')

parser.add_argument('--num_adj', type=int, default=9, help='number of nodes in sub graph')
parser.add_argument('--num_layer', type=int, default=2, help='number of layers in LSTM and DCRNN')
parser.add_argument('--trend_time', type=int, default=7 * 24, help='the length of trend segment is 7 days')

parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda_id', type=str, default='3')
parser.add_argument('--seed', type=int, default=20)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if not args.cuda:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# parameter
opt = vars(args)
# 2017-01-01 - 2017-05-06
if opt['dataset'] == 'bay':
    opt['timestamp'] = 12       # 5min: 12 or 30min: 2
    opt['train_time'] = 105     # days for training 
    opt['recent_time'] = 1      # bay: 1 hour, nyc: 2hour
    opt['num_feature'] = 6 * 2      # length of input feature
    opt['time_feature'] = 31        # length of time feature
# 2014-01-15 -- 2017-12-31
elif opt['dataset'] == 'nyc':
    opt['timestamp'] = 2       # 5min: 12 or 30min: 2
    opt['train_time'] = 289     # days for training
    opt['recent_time'] = 2      # bay: 1 hour, nyc: 2hour
    opt['num_feature'] = 2 * 2      # length of input feature
    opt['time_feature'] = 39        # length of time feature
elif opt['dataset'] == 'ximantis':
    opt['timestamp'] = 60       # 1min: 60
    opt['train_time'] = 8     # days for training
    opt['recent_time'] = 1      # ximantis: 1 hour
    opt['num_feature'] = 1 * 1      # length of input feature, density
    opt['time_feature'] = 24       # length of time feature, 24
elif opt['dataset'] == 'ximantis_smooth':
    opt['timestamp'] = 12       # 1min: 60
    opt['train_time'] = 12     # days for training
    opt['recent_time'] = 1      # ximantis: 1 hour
    opt['num_feature'] = 1      # length of input feature, density
    opt['time_feature'] = 24       # length of time feature, 24
elif opt['dataset'] == 'ximantis_smooth_2':
    opt['timestamp'] = 12       # 5min: 12
    opt['train_time'] = 161     # days for training: 161/181
    opt['recent_time'] = 1      # ximantis: 1 hour
    opt['num_feature'] = 1      # length of input feature, density
    opt['time_feature'] = 31       # length of time feature, 24 + 7
elif opt['dataset'] == 'ximantis_smooth_3':
    opt['timestamp'] = 12       # 5min: 12
    opt['train_time'] = 227     # days for training: 227/237
    opt['recent_time'] = 1      # ximantis: 1 hour
    opt['num_feature'] = 1      # length of input feature, density
    opt['time_feature'] = 31       # length of time feature, 24 + 7
elif opt['dataset'] == 'ximantis_smooth_3_truncated':
    opt['timestamp'] = 12       # 5min: 12
    opt['train_time'] = 227     # days for training: 227/237, but (252-58)/288 * 227 = 152.909722222
    opt['truncated'] = True
    opt['truncation_t1'] = 58   # truncate before 53/288, and after 252/288
    opt['truncation_t2'] = 252
    opt['recent_time'] = 1      # ximantis: 1 hour
    opt['num_feature'] = 1      # length of input feature, density
    opt['time_feature'] = 31       # length of time feature, 24 + 7

opt['save_path'] = opt['root_path'] + opt['dataset'] + '/checkpoint/'
opt['data_path'] = opt['root_path'] + opt['dataset'] + '/data/'
opt['result_path'] = opt['root_path'] + opt['dataset'] + '/result/'

current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
opt['result_path'] = opt['result_path'] + current_datetime + "/"
opt['save_path'] = opt['save_path'] + current_datetime + "/"

if opt.get('truncated') == True:
    correction = opt['truncation_t2'] - opt['truncation_t1']
    opt['train_time'] = opt['train_time'] * correction
else:
    opt['train_time'] = opt['train_time'] * opt['timestamp'] * 24

if __name__ == "__main__":
    logger.info("configuration:")
    logger.info(str(opt))

    opt['isTrain'] = True
    train_model = Trainer(opt)
    train_model.train()

    opt['isTrain'] = False
    test_model = Tester(opt)
    test_model.test()

    logger.info('completed successfully')
