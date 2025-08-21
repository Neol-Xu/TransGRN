import scipy.sparse as ssp
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
import scipy.sparse as ssp
import pandas as pd
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

class Pair:
    def __init__(self, TF_embed, Target_embed, link_label, cell_line_label):
        self.TF_embed = TF_embed
        self.Target_embed = Target_embed
        self.link_label = link_label
        self.cell_line_label = cell_line_label

def extract_cell_line_data(cell_line, data_type, network, num_genes, sample, ratio = None, target_cell_line = None):
    if data_type == "pre_train":
        data_path = f'demo_data/New_source_data/{network}/TFs+{num_genes}/{target_cell_line}/{cell_line}/Train_set.csv'
    
    # benchmark experiment
    elif data_type == "train":
        data_path = f'demo_data/benchmark/{network}/{cell_line}_{num_genes}/{sample}/Train_set.csv'
    elif data_type == "Validation":
        data_path = f'demo_data/benchmark/{network}/{cell_line}_{num_genes}/{sample}/Validation_set.csv'
    elif data_type == "test":
        data_path = f'demo_data/benchmark/{network}/{cell_line}_{num_genes}/{sample}/Test_set.csv'    
    
    # Few-shot experiment
    elif data_type == "few_shot_train":
        data_path = f'demo_data/Fewshot/{ratio}/{network}/{cell_line}_{num_genes}/{sample}/Train_set.csv'
    elif data_type == "few_shot_test":
        data_path = f'demo_data/Fewshot/{ratio}/{network}/{cell_line}_{num_genes}/{sample}/Test_set.csv'
    else:
        raise ValueError("data_type error")

    expression_path = f'demo_data/Expression/{network}/{cell_line}/TFs+{num_genes}/BL--ExpressionData.csv'
    gpt_path = f'demo_data/Aligned_Embeddings/{network}/{cell_line}/TFs+{num_genes}/Aligned_GPT_3_5_gene_embeddings.csv'

    # load data
    data = pd.read_csv(data_path, header=None, skiprows=1, usecols=[1, 2, 3])
    exp_data = pd.read_csv(expression_path, header=None, skiprows=1).iloc[:, 1:].values
    gpt_data = pd.read_csv(gpt_path, header=None, skiprows=1).iloc[:, 1:].values

    # standardize the expression and GPT data
    scaler = StandardScaler()
    exp_data = scaler.fit_transform(exp_data)
    gpt_data = scaler.fit_transform(gpt_data)

    # TruncatedSVD 
    svd = TruncatedSVD(n_components=256)
    exp_data_svd = svd.fit_transform(exp_data)

    exp_tensor = torch.tensor(exp_data_svd, dtype=torch.float32)
    gpt_tensor = torch.tensor(gpt_data, dtype=torch.float32)

    embedding_tensor = torch.cat((exp_tensor, gpt_tensor), dim=-1)
    links = (data.iloc[:, 0].astype(int), data.iloc[:, 1].astype(int))
    labels = data.iloc[:, 2]

    pairs = [
        Pair(
            TF_embed=embedding_tensor[TF_index], 
            Target_embed=embedding_tensor[Target_index],
            link_label=labels.iloc[label_index],
            cell_line_label=cell_line
        )
        for label_index, (TF_index, Target_index) in enumerate(zip(links[0], links[1]))
    ]

    return pairs

def load_cell_line_embeddings(file_path='demo_data/cell_lines_description.csv'):
    """
    Load cell line embeddings from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing cell line data.

    Returns:
        dict: A dictionary where keys are cell line names and values are torch tensors of features.
    """
    cell_line_data = pd.read_csv(file_path, header=None, skiprows=1)  # Read CSV file
    cell_line_embeddings = {}

    for idx, row in cell_line_data.iterrows():
        cell_line_name = row[0]  # First column is the cell line name
        cell_line_features = row[1:].astype(float).values  # Remaining columns are feature data
        cell_line_embeddings[cell_line_name] = torch.tensor(cell_line_features, dtype=torch.float32)  # Convert to torch tensor

    return cell_line_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate model on cell line data.")
    
    parser.add_argument("--network", type=str, help="Network directory name (e.g., 'Specific').")
    parser.add_argument("--num", type=str,help="Number of genes (e.g., '1000').")
    parser.add_argument("--sample", type=str, help="Sample directory name (e.g., 'sample1').")
    parser.add_argument("--source_cell_lines", type=str, required=True, help="List of source cell lines (e.g., 'hHEP mDC mESC mHSC-E mHSC-GM mHSC-L').")
    parser.add_argument("--target_cell_line", type=str, help="Target cell line (e.g., 'hESC').")
    parser.add_argument("--no-pretrain", action="store_false", dest="pretrain", help="Do not perform pretraining.")
    parser.add_argument("--no-finetune", action="store_false", dest="finetune", help="Do not perform finetuning.")
    parser.set_defaults(pretrain=True, finetune=True)  
    return parser.parse_args()