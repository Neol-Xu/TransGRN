import os
import pandas as pd
import numpy as np
import random

# Function to create the train set from label and gene/TF data
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

def generate_source_data():
    root_dir = os.getcwd()

    nets = ['Specific', 'Non-Specific', 'STRING']
    nums = ['500', '1000']
    cell_lines = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

    for net in nets:
        for num in nums:
            for target_cell in cell_lines:
                TF2file = os.path.join(root_dir, "demo_data\Expression", net, target_cell, f'TFs+{num}', 'TF.csv')
                Gene2file = os.path.join(root_dir, "demo_data\Expression", net, target_cell, f'TFs+{num}', 'Target.csv')
                label_file = os.path.join(root_dir, "demo_data\Expression", net, target_cell, f'TFs+{num}', 'Label.csv')

                if not os.path.exists(label_file):
                    continue

                path = os.path.join(root_dir, 'Source_data', net, f'{target_cell}_{num}', 'sample1')
                np.random.seed(0)
                random.seed(0)

                if not os.path.exists(path):
                    os.makedirs(path)

                train_set_file = os.path.join(path, 'Train_set.csv')

                train_set(label_file, Gene2file, TF2file, train_set_file)
                print(f'[SAVE] {train_set_file}')

# Function to check for duplicates and create New_Source dataset
def check_and_create_new_source():
    root_dir = os.getcwd()

    nets = ['Specific', 'Non-Specific', 'STRING']
    nums = ['500', '1000']
    cell_lines = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    out_root = os.path.join(root_dir, 'New_source_data')

    for net in nets:
        for num in nums:
            for target_cell in cell_lines:
                expr_t = os.path.join(
                    root_dir, 'demo_data/Expression', net, target_cell, f'TFs+{num}', 'Target.csv'
                )
                if not os.path.exists(expr_t):
                    continue
                df_map_t = pd.read_csv(expr_t)
                idx2gene_t = dict(zip(df_map_t['index'].astype(int), df_map_t['Gene'].astype(str)))

                # Collect test triplets (tf_gene, tg_gene, label)
                test_triplets = set()
                for i in range(1, 6):
                    test_file = os.path.join(
                        root_dir, 'benchmark', net, f'{target_cell}_{num}', f'sample{i}', 'Test_set.csv'
                    )
                    df_test = pd.read_csv(test_file)
                    for _, row in df_test.iterrows():
                        ti, gi = int(row['TF']), int(row['Target'])
                        lv = str(row['Label']).strip()
                        if ti in idx2gene_t and gi in idx2gene_t:
                            tf_gene, tg_gene = idx2gene_t[ti], idx2gene_t[gi]
                            test_triplets.add((tf_gene, tg_gene, lv))

                for source_cell in [cl for cl in cell_lines if cl != target_cell]:
                    expr_s = os.path.join(
                        root_dir, 'demo_data/Expression', net, source_cell, f'TFs+{num}', 'Target.csv'
                    )
                    df_map_s = pd.read_csv(expr_s)
                    idx2gene_s = dict(zip(df_map_s['index'].astype(int), df_map_s['Gene'].astype(str)))

                    # Prepare train set
                    in_csv = os.path.join(
                        root_dir, 'Source_data', net, f'{source_cell}_{num}', 'sample1', 'Train_set.csv'
                    )
                    df_train = pd.read_csv(in_csv)

                    keep = []
                    for _, row in df_train.iterrows():
                        if 'TF' in row and 'Target' in row and 'Label' in row:
                            try:
                                ti, gi = int(row['TF']), int(row['Target'])
                                lv = str(row['Label']).strip()
                                if ti in idx2gene_s and gi in idx2gene_s:
                                    tf_gene, tg_gene = idx2gene_s[ti], idx2gene_s[gi]
                                    triplet = (tf_gene, tg_gene, lv)
                                    keep.append(triplet not in test_triplets)
                                else:
                                    keep.append(True)
                            except Exception:
                                keep.append(True)
                        else:
                            keep.append(True)

                    df_out = df_train[keep].copy()

                    # Output directory and saving the train set
                    out_dir = os.path.join(out_root, net, f'TFs+{num}', target_cell, source_cell)
                    os.makedirs(out_dir, exist_ok=True)
                    out_csv = os.path.join(out_dir, 'Train_set.csv')
                    df_out.to_csv(out_csv, index=False, encoding='utf-8')

                    print(f'[SAVE] {target_cell} <- {source_cell} {net} TFs+{num} → {out_csv}')

# Run the process
if __name__ == '__main__':
    generate_source_data()  # First generate the single source dataset
    check_and_create_new_source()  # Then check and generate the New_Source dataset
