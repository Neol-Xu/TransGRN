import os
import pandas as pd

current_script_path = os.path.dirname(os.path.abspath(__file__))

base_dirs = [
    'Specific',
    'Non-specific',
    'STRING'
]
sub_dirs = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
sub_sub_dirs = ['TFs+1000', 'TFs+500']

species_gene_summary_human = set()
species_gene_summary_mouse = set()

for base_dir in base_dirs:
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for sub_sub_dir in sub_sub_dirs:
                sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
                if os.path.isdir(sub_sub_dir_path):
                    csv_file_path = os.path.join(sub_sub_dir_path, 'BL--ExpressionData.csv')
                    if os.path.exists(csv_file_path):
                        print(os.path.join(current_script_path, csv_file_path))
                        df = pd.read_csv(csv_file_path)
                        gene_names = df.iloc[:, 0].tolist()  

                        for gene_name in gene_names:
                            if sub_dir in ['hESC', 'hHEP']:
                                species_gene_summary_human.add(gene_name)
                            else:
                                species_gene_summary_mouse.add(gene_name)

output_file_human = 'human_gene_summary.csv'
df_output_human = pd.DataFrame(list(species_gene_summary_human), columns=['Gene Name'])
df_output_human.to_csv(output_file_human, index=False, encoding='utf-8-sig')

output_file_mouse = 'mouse_gene_summary.csv'
df_output_mouse = pd.DataFrame(list(species_gene_summary_mouse), columns=['Gene Name'])
df_output_mouse.to_csv(output_file_mouse, index=False, encoding='utf-8-sig')
