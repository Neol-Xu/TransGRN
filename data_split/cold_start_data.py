import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def Network_Statistic(data_type, net_scale, net_type):
    if net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}
        query = data_type + str(net_scale)
        return dic[query]
    else:
        raise ValueError("Unknown net_type")

def quantile_stratified_split(tf_network, 
                              train_size=0.6, val_size=0.2, test_size=0.2,
                              K=3, random_seed=42):
    """
    Input:
      tf_network: dict[str, list[str]]  # TF -> targets
    Output:
      (train_tfs, val_tfs, test_tfs)
    """
    rng = np.random.RandomState(random_seed)
    tfs = list(tf_network.keys())
    degrees = np.array([len(tf_network[tf]) for tf in tfs])

    qs = np.unique(np.quantile(degrees, q=np.linspace(0, 1, K+1)))
    bins = np.digitize(degrees, qs[1:-1], right=True)  # 0..K-1

    train_tfs, val_tfs, test_tfs = [], [], []
    for k in range(len(qs)-1):
        idx_k = np.where(bins == k)[0]
        tfs_k = [tfs[i] for i in idx_k]
        if len(tfs_k) == 0:
            continue

        train_k, temp_k = train_test_split(
            tfs_k, test_size=(1 - train_size), random_state=random_seed
        )
        if len(temp_k) > 0 and (val_size + test_size) > 0:
            test_ratio = test_size / (val_size + test_size)
            val_k, test_k = train_test_split(
                temp_k, test_size=test_ratio, random_state=random_seed
            )
        else:
            val_k, test_k = [], []

        train_tfs.extend(train_k)
        val_tfs.extend(val_k)
        test_tfs.extend(test_k)

    return train_tfs, val_tfs, test_tfs

def preassign_targets_disjoint(tf_network, tf_splits, ratios=(0.6, 0.2, 0.2), random_seed=42):
    """
    Pre-assign all targets to three mutually exclusive pools: train, val, and test.
    - tf_splits: (train_tfs, val_tfs, test_tfs)
    - ratios: Desired edge count ratios to determine the approximate "positive edge" quota for each set.
    Strategy: Traverse targets in descending order of degree (number of TFs pointing to them),
              and greedily assign each target to the set with the largest current "deficit."
              However, prioritize sets that have connections with the target's TFs (to produce positive edges).
    """
    random.seed(random_seed)

    train_tfs, val_tfs, test_tfs = tf_splits
    tf2set = {}
    for tf in train_tfs: tf2set[tf] = 0
    for tf in val_tfs:   tf2set[tf] = 1
    for tf in test_tfs:  tf2set[tf] = 2

    target_neighbors = defaultdict(list)
    total_pos_edges = 0
    for tf, tgts in tf_network.items():
        for t in tgts:
            target_neighbors[t].append(tf)
            total_pos_edges += 1

    targets = list(target_neighbors.keys())

    target_edges_quota = [ratios[i] * total_pos_edges for i in range(3)]
    edges_assigned = [0, 0, 0]

    def pot_edges_by_set(t):
        cnt = [0, 0, 0]
        for tf in target_neighbors[t]:
            s = tf2set.get(tf, None)
            if s is not None:
                cnt[s] += 1
        return cnt  # [train_cnt, val_cnt, test_cnt]

    targets.sort(key=lambda t: len(target_neighbors[t]), reverse=True)

    pools = [set(), set(), set()]  # 0:train, 1:val, 2:test

    for t in targets:
        pot = pot_edges_by_set(t)
        deficits = [target_edges_quota[i] - edges_assigned[i] for i in range(3)]
        cand = list(range(3))
        cand.sort(key=lambda i: (pot[i] == 0, -deficits[i]))
        chosen = cand[0]

        pools[chosen].add(t)
        edges_assigned[chosen] += pot[chosen]

    train_targets, val_targets, test_targets = pools
    return train_targets, val_targets, test_targets


def generate_edges_with_pools(tf_network, selected_tfs, target_pool, all_genes,
                              mode='train', density=None, neg_pos_ratio=1.0, random_seed=42):
    """
    Generate positive and negative samples only within the target_pool, ensuring target exclusivity across the three sets.
    - neg_pos_ratio: Ratio of negative samples to positive samples for train/val; for test, if density is provided, it is used to calculate the ratio.
    """
    random.seed(random_seed)
    pos_edges, neg_edges = [], []

    for tf in selected_tfs:
        tgts = tf_network.get(tf, [])
        for t in tgts:
            if t in target_pool:
                pos_edges.append((tf, t))

    tf_true = {tf: set(tf_network.get(tf, [])) for tf in selected_tfs}
    target_pool_list = list(target_pool)

    tf_pos_count = defaultdict(int)
    for tf, t in pos_edges:
        tf_pos_count[tf] += 1

    for tf in selected_tfs:
        p = tf_pos_count.get(tf, 0)
        if p == 0:
            continue
        if mode in ['train', 'val']:
            n_neg = int(round(p * neg_pos_ratio))
        else:
            n_neg = int(p * (1 - density) / density) if density and density > 0 else 0

        if n_neg <= 0:
            continue

        cand = [g for g in target_pool_list if g not in tf_true[tf]]
        if len(cand) == 0:
            continue
        sampled = random.sample(cand, min(n_neg, len(cand)))
        neg_edges.extend([(tf, g) for g in sampled])

    return pos_edges, neg_edges

def create_dataset(label_file, Gene_file, TF_file, output_dir, 
                   density, random_seed=42,
                   ratios=(0.6, 0.2, 0.2), neg_pos_ratio=1.0,
                   K=3):
    """
    Construct Train/Val/Test sets (TF-based split + mutually exclusive targets for cold start).
    - ratios: Desired positive edge ratios (guiding target pool allocation).
    - neg_pos_ratio: Negative-to-positive ratio for train/val (test is controlled by density).
    - K: Number of layers for stratification in quantile_stratified_split.
    """
    label = pd.read_csv(label_file)
    genes = pd.read_csv(Gene_file)['index'].tolist()
    tfs = pd.read_csv(TF_file)['index'].tolist()
    
    tf_network = defaultdict(list)
    for _, row in label.iterrows():
        tf_network[row['TF']].append(row['Target'])

    max_retry = 10
    for attempt in range(max_retry):
        seed = random_seed + attempt

        train_tfs, val_tfs, test_tfs = quantile_stratified_split(
            tf_network, K=K, random_seed=seed
        )

        train_targets, val_targets, test_targets = preassign_targets_disjoint(
            tf_network, (train_tfs, val_tfs, test_tfs), ratios=ratios, random_seed=seed
        )

        train_pos, train_neg = generate_edges_with_pools(
            tf_network, train_tfs, train_targets, genes,
            mode='train', neg_pos_ratio=neg_pos_ratio, random_seed=seed
        )
        val_pos, val_neg = generate_edges_with_pools(
            tf_network, val_tfs, val_targets, genes,
            mode='val', neg_pos_ratio=neg_pos_ratio, random_seed=seed
        )
        test_pos, test_neg = generate_edges_with_pools(
            tf_network, test_tfs, test_targets, genes,
            mode='test', density=density, random_seed=seed
        )

        if not train_pos or not val_pos or not test_pos:
            continue

        os.makedirs(output_dir, exist_ok=True)

        def to_df(pos, neg):
            df = pd.DataFrame(pos + neg, columns=['TF','Target'])
            df['Label'] = [1]*len(pos) + [0]*len(neg)
            return df

        train_df = to_df(train_pos, train_neg)
        val_df   = to_df(val_pos, val_neg)
        test_df  = to_df(test_pos, test_neg)

        train_df.to_csv(os.path.join(output_dir, 'Train_set.csv'))
        val_df.to_csv(os.path.join(output_dir, 'Validation_set.csv'))
        test_df.to_csv(os.path.join(output_dir, 'Test_set.csv'))

        train_tg, val_tg, test_tg = set(train_df['Target']), set(val_df['Target']), set(test_df['Target'])
        assert len(train_tg & val_tg) == 0, "Train-Val target overlap"
        assert len(train_tg & test_tg) == 0, "Train-Test target overlap"
        assert len(val_tg & test_tg) == 0,   "Val-Test target overlap"

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
            'unique_train_genes': len(train_tg),
            'unique_val_genes': len(val_tg),
            'unique_test_genes': len(test_tg),
            'is_cold_start': True
        }
        return stats

    raise RuntimeError("[ERROR] Constraints are too tight or data is extreme: Please increase the retry limit, relax ratios, increase neg_pos_ratio, or adjust K")

if __name__ == '__main__':
    data_types = ["hESC" , "hHEP", "mDC", "mHSC-E", "mESC", "mHSC-GM", "mHSC-L"]
    net_types = ["STRING"]
    nums = [1000]

    RATIOS = (0.6, 0.2, 0.2)  
    NEG_POS_RATIO = 1.0       
    K_LAYER = 3  # Number of TF layers (K for quantile_stratified_split)

    for net_type in net_types:
        for num in nums:
            for data_type in data_types:
                base_dir = os.getcwd()
                TF_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'TF.csv')
                Gene_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'Target.csv')
                label_file = os.path.join(base_dir, "demo_data/Expression", net_type, data_type, f'TFs+{num}', 'Label.csv')
                density = Network_Statistic(data_type, num, net_type)

                print(f"\nProcessing: {net_type}/{data_type}/TFs+{num} | Density: {density:.4f}")
                
                for i in range(1, 6):
                    output_dir = os.path.join(
                        base_dir, 'cold_start', 
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
                        random_seed=i,
                        ratios=RATIOS,
                        neg_pos_ratio=NEG_POS_RATIO,
                        K=K_LAYER
                    )
                    
                    print(f"Split {i} stats:")
                    print(f"  TFs: Train={stats['train_tfs']}, Val={stats['val_tfs']}, Test={stats['test_tfs']}")
                    print(f"  Train edges: Pos={stats['train_pos']}, Neg={stats['train_neg']} (Ratio: 1:{(stats['train_neg']/stats['train_pos'] if stats['train_pos'] else 0):.2f})")
                    print(f"  Val edges:   Pos={stats['val_pos']},   Neg={stats['val_neg']}   (Ratio: 1:{(stats['val_neg']/stats['val_pos'] if stats['val_pos'] else 0):.2f})")
                    print(f"  Test edges:  Pos={stats['test_pos']},  Neg={stats['test_neg']}  (Ratio: 1:{stats['test_ratio']:.2f}, Target: 1:{((1-density)/density if density>0 else 0):.2f})\n")
