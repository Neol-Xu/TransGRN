import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

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
def stratified_tf_split(tf_network, test_size=0.4, val_size=0.5, random_seed=42):
    tiers = {
        'tier1': [],  
        'tier2': [],  
        'tier3': [],  
        'tier4': []   
    }
    
    for tf, targets in tf_network.items():
        n_targets = len(targets)
        if n_targets > 500:
            tiers['tier1'].append(tf)
        elif n_targets > 100:
            tiers['tier2'].append(tf)
        elif n_targets > 10:
            tiers['tier3'].append(tf)
        else:
            tiers['tier4'].append(tf)
    
    train_tfs, val_tfs, test_tfs = [], [], []
    
    for tier, tfs in tiers.items():
        if tier == 'tier4':
            test_tfs.extend(tfs)
            continue
            
        if len(tfs) <= 3: 
            if len(tfs) == 1:
                train_tfs.extend(tfs)   
            elif len(tfs) == 2:
                train_tfs.append(tfs[0])
                val_tfs.append(tfs[1])  
            elif len(tfs) == 3:
                train_tfs.append(tfs[0])
                val_tfs.append(tfs[1])  
                test_tfs.append(tfs[2]) 
            continue
            
        train, temp = train_test_split(
            tfs, 
            test_size=test_size, 
            random_state=random_seed
        )
        val, test = train_test_split(
            temp, 
            test_size=val_size, 
            random_state=random_seed
        )
        
        train_tfs.extend(train)
        val_tfs.extend(val)
        test_tfs.extend(test)
    
    return train_tfs, val_tfs, test_tfs

def generate_edges(tf_network, selected_tfs, all_genes, used_genes, used_targets, mode='train', density=None, random_seed=42):
    random.seed(random_seed)
    pos_edges = []
    neg_edges = []
    
    for tf in selected_tfs:
        targets = tf_network.get(tf, [])
        available_targets = [t for t in targets if t not in used_targets]
        pos_edges.extend([(tf, t) for t in available_targets])
        
        if mode in ['train', 'val']:
            n_neg = len(available_targets) 
        else:
            n_pos = len(available_targets)
            n_neg = int(n_pos * (1 - density) / density) if density > 0 else 0
        
        possible_neg = [g for g in all_genes 
                       if g not in targets and g not in used_genes and g not in used_targets]
        
        if n_neg > 0 and possible_neg:
            sampled_neg = random.sample(possible_neg, min(n_neg, len(possible_neg)))
            neg_edges.extend([(tf, g) for g in sampled_neg])
    
    return pos_edges, neg_edges

def create_dataset(label_file, Gene_file, TF_file, output_dir, 
                   density, random_seed=42):
    label = pd.read_csv(label_file)
    genes = pd.read_csv(Gene_file)['index'].tolist()
    tfs = pd.read_csv(TF_file)['index'].tolist()
    
    tf_network = defaultdict(list)
    for _, row in label.iterrows():
        tf_network[row['TF']].append(row['Target'])
    max_retry = 10
    for attempt in range(max_retry):
        random_seed = random_seed + attempt
        train_tfs, val_tfs, test_tfs = stratified_tf_split(
            tf_network, 
            random_seed=random_seed
        )
        
        used_genes = set()  
        used_targets = set()
        
        train_pos, train_neg = generate_edges(
            tf_network, train_tfs, genes, 
            used_genes, used_targets,
            mode='train', random_seed=random_seed
        )
        if not train_pos or not train_neg:
            continue
        train_df = pd.DataFrame(
            train_pos + train_neg,
            columns=['TF', 'Target']
        )
        train_df['Label'] = [1]*len(train_pos) + [0]*len(train_neg)
        
        used_genes.update(train_tfs)
        used_targets.update([t for _, t in train_pos + train_neg])
        
        val_pos, val_neg = generate_edges(
            tf_network, val_tfs, genes,
            used_genes, used_targets,
            mode='val', random_seed=random_seed
        )
        if not val_pos or not val_neg:
            continue
        val_df = pd.DataFrame(
            val_pos + val_neg,
            columns=['TF', 'Target']
        )
        val_df['Label'] = [1]*len(val_pos) + [0]*len(val_neg)
        
        used_genes.update(val_tfs)
        used_targets.update([t for _, t in val_pos + val_neg])
        
        test_pos, test_neg = generate_edges(
            tf_network, test_tfs, genes,
            used_genes, used_targets,
            mode='test', density=density, random_seed=random_seed
        )
        if not test_pos or not test_neg:
            continue

        test_df = pd.DataFrame(
            test_pos + test_neg,
            columns=['TF', 'Target']
        )
        test_df['Label'] = [1]*len(test_pos) + [0]*len(test_neg)
        
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'Train_set.csv'))
        val_df.to_csv(os.path.join(output_dir, 'Validation_set.csv'))
        test_df.to_csv(os.path.join(output_dir, 'Test_set.csv'))
        train_tg = set(train_df['Target'])
        val_tg = set(val_df['Target'])
        test_tg = set(test_df['Target'])

        assert len(train_tg & val_tg) == 0, "Train Val overlap in Target genes"
        assert len(train_tg & test_tg) == 0, "Train Test overlap in Target genes"
        assert len(val_tg & test_tg) == 0, "Val Test overlap in Target genes"
        
        stats = {
            'total_tfs': len(tf_network),
            'train_tfs': len(train_tfs),
            'val_tfs': len(val_tfs),
            'test_tfs': len(test_tfs),
            'train_pos': len(train_pos),
            'train_neg': len(train_neg),
            'val_pos': len(val_pos),
            'val_neg': len(val_neg),
            'test_pos': len(test_pos),
            'test_neg': len(test_neg),
            'test_ratio': len(test_neg)/len(test_pos) if len(test_pos) > 0 else 0,
            'unique_train_genes': len(set([t for _, t in train_pos + train_neg])),
            'unique_val_genes': len(set([t for _, t in val_pos + val_neg])),
            'unique_test_genes': len(set([t for _, t in test_pos + test_neg])),
            'is_cold_start': len(set([t for _, t in train_pos + train_neg]) & 
                            set([t for _, t in test_pos + test_neg])) == 0
        }
        return stats
    raise RuntimeError("[ERROR]")

if __name__ == '__main__':
    data_types = ["hESC", "hHEP", "mDC", "mHSC-E", "mESC", "mHSC-GM", "mHSC-L"]
    net_types = ["STRING"]
    nums = [1000]
    
    for net_type in net_types:
        for num in nums:
            for data_type in data_types:

                base_dir = os.getcwd()
                TF_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'TF.csv')
                Gene_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'Target.csv')
                label_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'Label.csv')
                density = Network_Statistic(data_type, num, net_type)

                print(f"\nProcessing: {net_type}/{data_type}/TFs+{num} | Density: {density:.4f}")
                
                for i in range(1,6):
                    output_dir = os.path.join(
                        base_dir, 'cold_start_V4', 
                        net_type, 
                        f'{data_type}_{num}', 
                        f'sample{i}'
                    )
                    
                    stats = create_dataset(
                        label_file=label_file,
                        Gene_file=Gene_file,
                        TF_file=TF_file,
                        output_dir=output_dir,
                        density=density,
                        random_seed=i
                    )
                    
                    print(f"Split {i} stats:")
                    print(f"  TFs: Train={stats['train_tfs']}, Val={stats['val_tfs']}, Test={stats['test_tfs']}")
                    print(f"  Train edges: Pos={stats['train_pos']}, Neg={stats['train_neg']} (Ratio: 1:{stats['train_neg']/stats['train_pos']:.2f})")
                    print(f"  Val edges: Pos={stats['val_pos']}, Neg={stats['val_neg']} (Ratio: 1:{stats['val_neg']/stats['val_pos']:.2f})")
                    print(f"  Test edges: Pos={stats['test_pos']}, Neg={stats['test_neg']} (Ratio: 1:{stats['test_ratio']:.2f}, Target: 1:{(1-density)/density:.2f})\n")