import os
import pandas as pd

def check_and_create_new_source():
    root_dir = os.getcwd()

    nets = ['STRING']
    nums = ['1000']
    cell_lines = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    out_root = os.path.join(root_dir, 'Cold_start_source_data')

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

                test_tfs = set()
                test_targets = set()
                for i in range(1, 6):
                    test_file = os.path.join(
                        root_dir, 'cold_start', net, f'{target_cell}_{num}', f'sample{i}', 'Test_set.csv'
                    )
                    df_test = pd.read_csv(test_file)
                    for _, row in df_test.iterrows():
                        ti, gi = int(row['TF']), int(row['Target'])
                        if ti in idx2gene_t and gi in idx2gene_t:
                            tf_gene, tg_gene = idx2gene_t[ti], idx2gene_t[gi]
                            test_tfs.add(tf_gene)  # Collect test TFs
                            test_targets.add(tg_gene)  # Collect test Targets

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
                            ti, gi = int(row['TF']), int(row['Target'])
                            if ti in idx2gene_s and gi in idx2gene_s:
                                tf_gene, tg_gene = idx2gene_s[ti], idx2gene_s[gi]
                                if tf_gene in test_tfs or tg_gene in test_targets:
                                    keep.append(False)
                                else:
                                    keep.append(True)
                            else:
                                keep.append(False)

                    df_out = df_train[keep].copy()

                    # Output directory and saving the train set
                    out_dir = os.path.join(out_root, net, f'TFs+{num}', target_cell, source_cell)
                    os.makedirs(out_dir, exist_ok=True)
                    out_csv = os.path.join(out_dir, 'Train_set.csv')
                    df_out.to_csv(out_csv, index=False, encoding='utf-8')

                    print(f'[SAVE] {target_cell} <- {source_cell} {net} TFs+{num} → {out_csv}')

if __name__ == '__main__':
    check_and_create_new_source()  # Then check and generate the New_Source dataset
