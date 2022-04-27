import torch.utils.data as tud
import random
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
from utils import arr2hot
from tqdm import tqdm


class ScoreDataSet(tud.Dataset):
    def __init__(self, data):
        super(ScoreDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return int(self.data[item]['sid']), int(self.data[item]['pid']), torch.Tensor(self.data[item]['Q']), float(self.data[item]['rt']), self.data[item]['r']

class ScoreDataLoader(object):
    def __init__(self, dataset_file, batch_size, train_rate, K):
        print("loading data...")
        data, n_stu, n_pro, n_know = self.load_data(dataset_file)
        train_data, test_data = self.init_data(data, train_rate, n_know)
        train_val_data = self.get_KFold_data(train_data, K)
        test_dataSet = ScoreDataSet(test_data)
        self.train_val_dataloader = [{'train': tud.DataLoader(tv['train'], batch_size=batch_size, shuffle=True, num_workers=0),
                                  'val': tud.DataLoader(tv['val'], batch_size=batch_size, shuffle=True, num_workers=0)}
                                 for tv in train_val_data]
        self.test_dataloader = tud.DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, num_workers=0)
        self.n_stu, self.n_pro, self.n_know = n_stu, n_pro, n_know

    def get_KFold_data(self, data, K):
        kf = KFold(n_splits=K)
        train_val_data = []
        for train_index, val_index in kf.split(data):
            train_data = [data[i] for i in train_index]
            val_data = [data[i] for i in val_index]
            train_val_data.append({'train': ScoreDataSet(train_data), 'val': ScoreDataSet(val_data)})
        return train_val_data

    def load_data(self, dataset_file):
        print('using dataset: ' + dataset_file)
        if dataset_file == 'pisa2015':
            data = pd.read_pickle('./data/pisa2015.pickle')
            n_stu, n_pro, n_know = 1476, 17, 11
        elif dataset_file == 'junyi':
            data = pd.read_pickle('./data/junyi.pickle')
            n_stu, n_pro, n_know = 15000, 718, 39
        elif dataset_file == 'ednet':
            data = pd.read_pickle('./data/ednet.pickle')
            n_stu, n_pro, n_know = 9980, 12165, 188
        elif dataset_file == 'assistment2017':
            data = pd.read_pickle('./data/assistment2017.pickle')
            data = data[data['rt'] <= 100].reset_index(drop=True)
            n_stu, n_pro, n_know = 1709, 3162, 102
        else:
            print('No available dataset ...')
            exit(-1)
        return data, n_stu, n_pro, n_know


    def init_data(self, data, train_rate, n_know):
        No = data.shape[0]
        idx = np.random.permutation(No)
        Train_No = int(No * train_rate)

        train_data = []
        test_data = []
        for i in tqdm(range(Train_No)):
            train_data.append({'sid': data.loc[idx[i], 'sid'], 'pid': data.loc[idx[i], 'pid'],
                               'Q': arr2hot(data.loc[idx[i], 'Q'], n_know), 'rt': data.loc[idx[i], 'rt'], 'r':
                               data.loc[idx[i], 'r']})
        for i in tqdm(range(Train_No, No)):
            test_data.append({'sid': data.loc[idx[i], 'sid'], 'pid': data.loc[idx[i], 'pid'],
                               'Q': arr2hot(data.loc[idx[i], 'Q'], n_know), 'rt': data.loc[idx[i], 'rt'], 'r':
                               data.loc[idx[i], 'r']})
        return train_data, test_data

    def init_data1(self, data, train_rate, n_know):
        stu_map = {}
        train_data = []
        test_data = []
        data = data.sample(frac=1)
        print('total records num: {}'.format(data.shape[0]))
        for index, row in tqdm(data.iterrows()):
            sid = row['sid']
            if sid not in stu_map.keys():
                stu_map[sid] = []
            stu_map[sid].append(index)
        for sid in tqdm(stu_map):
            l = len(stu_map[sid])
            th = round(l * train_rate)
            for i in range(l):
                record = {'sid': data.loc[stu_map[sid][i], 'sid'], 'pid': data.loc[stu_map[sid][i], 'pid'],
                          'Q': arr2hot(data.loc[stu_map[sid][i], 'Q'], n_know), 'rt': data.loc[stu_map[sid][i], 'rt'],
                          'r': data.loc[stu_map[sid][i], 'r']}
                if i < th:
                    train_data.append(record)
                else:
                    test_data.append(record)
        print(len(train_data))
        print(len(test_data))
        return train_data, test_data





    def get_train_val_dataloader(self):
        return self.train_val_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_data_shape(self):
        return self.n_stu, self.n_pro, self.n_know
