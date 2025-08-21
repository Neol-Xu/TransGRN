import os
import pandas as pd
import numpy as np

def load_target_index_map(target_path):
    df = pd.read_csv(target_path)
    return dict(zip(df['index'], df['Gene'])), dict(zip(df['Gene'], df['index']))

def negative_sampling(positive_edges, all_TFs, candidate_targets, sample_num):
    neg_edges = set()
    while len(neg_edges) < sample_num:
        tf = np.random.choice(list(all_TFs))
        target = np.random.choice(list(candidate_targets))
        if (tf, target) not in positive_edges and (tf, target) not in neg_edges:
            neg_edges.add((tf, target))
    return list(neg_edges)

test_base_dir = "/data/cold_start/STRING"
gt_base_dir = "/data/Expression/STRING"

cell_lines = ["hESC", "hHEP", "mDC", "mHSC-E", "mESC", "mHSC-GM", "mHSC-L"]
samples = ["sample1", "sample2", "sample3", "sample4", "sample5"]
tf_nums = [1000]

for tf_num in tf_nums:
    for target_cl in cell_lines:
        for sample in samples:
            print(f"Processing Target cell line {target_cl}, TF count {tf_num}, sample {sample}")

            test_set_path = os.path.join(test_base_dir, f"{target_cl}_{tf_num}", sample, "Test_set.csv")
            test_df = pd.read_csv(test_set_path)

            target_path = f"/data/Expression/STRING/{target_cl}/TFs+{tf_num}/Target.csv"
            index2gene_target, _ = load_target_index_map(target_path)

            test_df['TF'] = test_df['TF'].map(index2gene_target)
            test_df['Target'] = test_df['Target'].map(index2gene_target)
            
            excluded_genes = set(test_df['TF']).union(set(test_df['Target']))

            source_cell_lines = [cl for cl in cell_lines if cl != target_cl]

            for source_cl in source_cell_lines:
                print(f"Processing Source cell line {source_cl}")

                gt_path = os.path.join(gt_base_dir, source_cl, f"TFs+{tf_num}", "BL--network.csv")
                gt_df = pd.read_csv(gt_path)

                filtered_gt_df = gt_df[
                    (~gt_df['Gene1'].isin(excluded_genes)) &
                    (~gt_df['Gene2'].isin(excluded_genes))
                ]

                positive_edges = set(zip(filtered_gt_df['Gene1'], filtered_gt_df['Gene2']))
                sample_num = len(positive_edges)

                source_path = f"/data/Expression/STRING/{source_cl}/TFs+{tf_num}/Target.csv"
                df = pd.read_csv(source_path)
                candidate_targets = set(df['Gene'].unique())
                candidate_targets = candidate_targets - excluded_genes

                all_TFs = set(filtered_gt_df['Gene1'].unique())
                neg_edges = negative_sampling(positive_edges, all_TFs, candidate_targets, sample_num)

                pos_df = pd.DataFrame(list(positive_edges), columns=['TF', 'Target'])
                pos_df['Label'] = 1
                neg_df = pd.DataFrame(neg_edges, columns=['TF', 'Target'])
                neg_df['Label'] = 0
                train_df = pd.concat([pos_df, neg_df])

                _, gene2index_source = load_target_index_map(source_path)
                train_df['TF'] = train_df['TF'].map(gene2index_source)
                train_df['Target'] = train_df['Target'].map(gene2index_source)

                save_dir = f"/data/cold_start_pretrain/{target_cl}/TFs_{tf_num}/{sample}/{source_cl}"
                os.makedirs(save_dir, exist_ok=True)
                train_df.to_csv(os.path.join(save_dir, "Train_set.csv"))