import pandas as pd
import numpy as np
import os
import numpy as np
import random

def Network_Statistic(data_type, net_scale, net_type):

    if net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165, 'hHEP500': 0.379, 'hHEP1000': 0.377, 'mDC500': 0.085,
               'mDC1000': 0.082, 'mESC500': 0.345, 'mESC1000': 0.347, 'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565, 'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError

def train_set(label_file, Gene_file, TF_file, train_set_file):
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf_in_label = label['TF'].unique()
    
    all_pos_edges = set(map(tuple, label[['TF', 'Target']].values))
    all_possible_edges = {(t1, t2) for t1 in tf_in_label for t2 in gene_set}
    candidate_neg_edges = list(all_possible_edges - all_pos_edges)
    
    if len(all_pos_edges) > len(candidate_neg_edges):
        train_pos_edges = random.sample(list(all_pos_edges), len(candidate_neg_edges))
    else:
        train_pos_edges = list(all_pos_edges)
    
    train_neg_edges = random.sample(candidate_neg_edges, len(train_pos_edges))
    train_set = train_pos_edges + train_neg_edges
    train_label = [1] * len(train_pos_edges) + [0] * len(train_neg_edges)
    
    train = pd.DataFrame(train_set, columns=['TF', 'Target'])
    train['Label'] = train_label
    train.to_csv(train_set_file)


if __name__ == '__main__':
    data_types = ["mHSC-E","hESC", "hHEP", "mDC", "mESC",  "mHSC-GM", "mHSC-L"]  
    net_types = ["Specific","Non-Specific","STRING"] 
    nums = [500, 1000] 

    for data_type in data_types:
        for net_type in net_types:
            for num in nums:

                TF2file = os.getcwd() + '/' + net_type + '/' + data_type + '/TFs+' + str(num) + '/TF.csv'
                Gene2file = os.getcwd() + '/'  + net_type + '/' + data_type + '/TFs+' + str(num) + '/Target.csv'
                label_file = os.getcwd() + '/'  + net_type + '/' + data_type + '/TFs+' + str(num) + '/Label.csv'
                density = Network_Statistic(data_type, num, net_type)
                if not os.path.exists(label_file):
                    continue

                for i in range(5):
                    path = os.getcwd() + '/Source_data/' + net_type + '/' + data_type + '_' + str(num) + '/sample' + str(i+1)
                    np.random.seed(i)
                    random.seed(i)

                    if not os.path.exists(path):
                        os.makedirs(path)

                    train_set_file = path + '/Train_set.csv'

                    train_set(label_file, Gene2file, TF2file, train_set_file)


