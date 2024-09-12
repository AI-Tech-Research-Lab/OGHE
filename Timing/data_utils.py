import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import structlog
import pandas as pd

log = structlog.get_logger()

PATH_NN = 'Dataset/'
cancer_list = ['Bladder', 'Breast', 'Bronchusandlung', 'Cervixuteri', 'Colon', 'Corpusuteri', 'Kidney', 'Liverandintrahepaticbileducts', 'Ovary', 'Skin', 'Stomach']
LOW = 0.2
MODERATE = 0.5
MODIFIER = 0.9
HIGH = 1
encodeSNPeff = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']

TRAIN_PATH_CNN = 'Dataset/numpy/TrainVal/'
TEST_PATH_CNN = 'Dataset/numpy/Test/'

VAL_SIZE = 270


def random_shuffle(data, label):
    if data.shape[1] != len(label):
        print('the number of samples does not match!')
        return
    indices = np.arange(len(label))
    np.random.shuffle(indices)

    return data.T[indices].T, label[indices]


def read_CN(path, sep = '\t', header_row = True, header_column = True):
    if header_row:
        data = np.array(pd.read_csv(path, sep = sep, header = 0))
    else:
        data = np.array(pd.read_csv(path, sep = sep, header = None))
    if header_column:
        data = data[:, 1:]
    return data.astype(np.float32)


def read_CNs(path_head, path_tail):
    data = []
    label = []
    for cancer in cancer_list:
        data.append(read_CN(path_head + cancer + path_tail, sep = '\t', header_row = True, header_column = True))
        label += [cancer_list.index(cancer)] * data[-1].shape[1]

    data = np.hstack(data)
    label = np.array(label)
    return data, label


def read_variants_csv(path_csv):
    SNPeff = np.array(pd.read_csv(path_csv, sep = '\t', header = None, dtype=float))
    SNPeff[SNPeff == 1] = MODIFIER
    SNPeff[SNPeff == 2] = LOW
    SNPeff[SNPeff == 3] = MODERATE
    SNPeff[SNPeff == 4] = HIGH
    return SNPeff


def read_all_data(cn_head, cn_tail, snpeff_csv):
    CN, true = read_CNs(cn_head, cn_tail)
    SNPeff = read_variants_csv(snpeff_csv)
    return CN, SNPeff, true


def freq(data):
  # temp = (data != 0)
  freq = np.sum(data, axis = 0)/data.shape[0]
  return freq


def freq_NN(N_SNV, data, labels):

    N = np.array([582, 1197, 2286, 4613])
    TR = np.array([127, 271, 530, 1140])

    idx_SNV = []

    if N_SNV in N:
        TR = TR[np.argwhere(N == N_SNV)][0][0]
        for i in range(11):
            freq = data[np.argmax(labels, axis = 1) == i].sum(axis = 0)
            idx_SNV = np.concatenate((idx_SNV, freq.argsort()[-TR:]))
            
        idx_SNV = np.unique([int(i) for i in idx_SNV])

    return idx_SNV


def dist(x, y):
    return 1 - np.sum(x == y) / (len(x) * 2. - np.sum(x == y))


def find_dist(temp):
  d = []
  for i in range(0, temp.shape[1]-1):
    d.append(dist(temp[:, i], temp[:, i+1]))
  return np.array(d)


def CNV_clustering(N_CNV, data):
  temp = np.concatenate(data).copy()
  d = find_dist(temp)
  idx_CNV = np.arange(0, temp.shape[1])
  dim = len(idx_CNV)

  while dim != N_CNV:
    if dim - len(np.argwhere(d == np.min(d))) > N_CNV or dim - len(np.argwhere(d == np.min(d))) == N_CNV:
        idx_CNV = np.delete(idx_CNV, np.argwhere(d == np.min(d))+1, 0)
    else:
      for j in range((dim - N_CNV)):
        idx_CNV = np.delete(idx_CNV, np.argwhere(d == np.min(d))[j]+1-j, 0)

    d = find_dist(temp[:, idx_CNV])
    dim = len(idx_CNV)

  return idx_CNV


def cluster(N_CNV, data):

  N = np.array([258, 710, 1024, 2031])
  TR = np.array([0.161, 0.0794, 0.061, 0.036])

  if N_CNV in N:
    TR = TR[np.argwhere(N == N_CNV)][0][0]
    CN = [0]
    for i in range(data.shape[1]):
        if dist(data[:, i], data[:, CN[-1]]) > TR:
            CN = np.concatenate((CN, [i]))
  else:
    CN = []
  return CN


# Feature reduction method
def feature_reduction(data, mode, N_CNV, N_SNV, train_labels):

  if mode == 'NN':
    CNV = data[0]
    SNV = data[1]
  else:
    CNV = []
    SNV = []

    for i in range(0, 11):
        CNV.append(data[i][:, 0, :])
        SNV.append(data[i][:, 1, :])

  ### CNV CLUSTERING ###
  if mode == 'NN':
      idx_CNV = cluster(N_CNV, CNV)
  elif mode == 'CNN':
      idx_CNV = CNV_clustering(N_CNV, CNV)
  else:
      print('No matching CNV mode!')
      idx_CNV = []

  print(f"CNV features:\t{len(idx_CNV)}")

  ### SNV FREQUENCY FILTER ###
  if mode == 'NN':
      idx_SNV = freq_NN(N_SNV, SNV, train_labels)
  elif mode == 'CNN':
      for i in range(0, 11):
          SNV[i][SNV[i] == 0.25] = 0.2
          SNV[i][SNV[i] == 0.75] = 0.9
      f_SNV = []
      ordered_f_SNV = []
      idx_SNV = []

      for i in range(0, len(SNV)):
          f_SNV.append(freq(SNV[i]))
          ordered_f_SNV.append(np.flip(np.argsort(f_SNV[i])))
      i = 0
      j = 0
      while len(idx_SNV) < N_SNV:
          idx_SNV = np.concatenate([idx_SNV, [ordered_f_SNV[j][i]]])
          idx_SNV = np.unique(idx_SNV)
          j = j + 1
          if j == 11:
              j = 0
              i = i + 1
  else:
      print('No matching mode!')
      idx_SNV = []

  print(f"SNV features:\t{len(idx_SNV)}")

  return idx_CNV, np.array(idx_SNV[:], dtype = int)


def preprocessing(data, test_data, label, test_label, N_CNV, N_SNV, mode):
  idx_CNV, idx_SNV = feature_reduction(data, mode, N_CNV, N_SNV, label)

  # Data filtering
  if mode == 'NN':
      x_train_CNV = data[0][:, idx_CNV]
      x_train_SNV = data[1][:, idx_SNV]

      x_test_CNV = test_data[0][:, idx_CNV]
      x_test_SNV = test_data[1][:, idx_SNV]
  else:
      x_train_CNV = np.concatenate(data)[:, 0, idx_CNV]
      x_train_SNV = np.concatenate(data)[:, 1, idx_SNV]

      x_test_CNV = np.concatenate(test_data)[:, 0, idx_CNV]
      x_test_SNV = np.concatenate(test_data)[:, 1, idx_SNV]

  train_data = np.concatenate([x_train_CNV, x_train_SNV], axis = 1)
  x_test = np.concatenate([x_test_CNV, x_test_SNV], axis = 1)

  # Train/val split
  x_train, x_val, y_train, y_val = train_test_split(train_data, label, test_size=270, random_state=1, stratify=label)
  y_test = test_label

  return x_train, y_train, x_val, y_val, x_test, y_test


def load_files(filename):
  pat = []
  temp = np.load(filename)
  for j in range(0, np.shape(temp)[2]):
    pat.append(temp[:, :, j])
  return np.array(pat)


def get_target(target, n_classes):
  exploded = np.zeros(shape=(len(target), n_classes), dtype=np.uint8)
  for i in range(0, len(target)):
    exploded[i, target[i]] = 1
  return exploded


def _get_data_dict(N_CNV:int, N_SNV:int, mode) -> dict:

    if mode == 'NN':

        # Train data
        CN, SNPeff, train_labels = read_all_data(PATH_NN + 'ds_train/',
                                         '_challenge_CNs.txt',
                                         PATH_NN + 'SNPeff/SNPeff_train.csv')
        train_data = [np.array(CN.T), np.array(SNPeff)]
        train_labels = get_target(train_labels, 11)

        # Test data
        CN_test, SNPeff_test, test_labels = read_all_data(PATH_NN + 'ds_test/',
                                                        '_evaluation_CNs.txt',
                                                        PATH_NN + 'SNPeff/SNPeff_test.csv')
        test_data = [CN_test.T, SNPeff_test]
        test_labels = get_target(test_labels, 11)

        # gene list
        gene_list = np.array(
            pd.read_csv(PATH_NN + 'SNPeff/variant_gene_list.csv', sep='\t',
                        header=None, dtype='str'))

    else:
        train_data = []
        train_labels = []

        for i in range(0, 11):
            file_name = 'Dataset/numpy/TrainVal/' + 'pat_' + str(i) + '.npy'
            train_data.append(load_files(file_name))
            if i == 0:
                train_labels.append(np.zeros(train_data[i].shape[0], dtype = int))
            else:
                train_labels.append(np.ones(train_data[i].shape[0], dtype = int)*i)

        train_labels = get_target(np.concatenate(train_labels), 11)

        # Loading test data
        test_data = []
        test_labels = []
        for i in range(0, 11):
            file_name = TEST_PATH_CNN + 'pat_' + str(i) + '.npy'
            test_data.append(load_files(file_name))
            if i == 0:
                test_labels.append(np.zeros(test_data[i].shape[0], dtype = int))
            else:
                test_labels.append(np.ones(test_data[i].shape[0], dtype = int)*i)

        test_labels = get_target(np.concatenate(test_labels), 11)

    # Preprocess the data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(train_data, test_data, train_labels, test_labels, N_CNV, N_SNV, mode)

    data_dict = {'train': {'x': x_train, 'y': y_train}, 
                  'val': {'x': x_val, 'y': y_val}, 
                  'test': {'x': x_test, 'y': y_test}}
        
    return data_dict


def create_stratified_folds(N_CNV:int, N_SNV:int, mode, batch_size:int, num_folds=5):
    data_dict = _get_data_dict(N_CNV, N_SNV, mode)

    x = np.concatenate((data_dict['train']['x'], data_dict['val']['x'], data_dict['test']['x']))
    y = np.concatenate((data_dict['train']['y'], data_dict['val']['y'], data_dict['test']['y']))

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(x, np.argmax(y, axis=1)):
        # Splitting the combined dataset into training/testing for each fold
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Within the training set, further split off a validation set (10% of the training set)
        val_size = int(0.1 * len(x_train))
        x_train, x_val = x_train[:-val_size], x_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]
        
        with open(f'sample_data_258_582.pkl', 'wb') as f:
                pickle.dump([x_test, y_test], f)

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        yield train_loader, val_loader, test_loader

def load_data_withoutFolds(N_CNV: int, N_SNV: int, mode):
        data_dict = _get_data_dict(N_CNV, N_SNV, mode)

        x_train = data_dict['train']['x']
        y_train = data_dict['train']['y']

        x_val = data_dict['val']['x']
        y_val = data_dict['val']['y']

        x_test = data_dict['test']['x']
        y_test = data_dict['test']['y']

        return x_train, y_train, x_val, y_val, x_test, y_test
