import os
import torch
import random
import os.path
import warnings
import datetime
import numpy as np
import scipy.sparse as ssp
import torch.optim as optim
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)

from main import *
from util_functions import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()

    num = args.num
    sample = args.sample
    network = args.network
    pretrain = args.pretrain
    target_cell_line = args.target_cell_line
    source_cell_lines = args.source_cell_lines.split(",")
    
    model = TransGRN().cuda() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  

    # Load cell line embeddings
    cell_line_embeddings = load_cell_line_embeddings()
    ###################### pre-training ######################
    if pretrain:
        print("######################")
        print("With Pre-training")
        print("######################")
        checkpoint_path = f'check_point/{network}/{target_cell_line}_{num}/sample1/model_weights.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded pre-trained model")
        else:
            Source_cell_line_queues = {}
            for cell_line in source_cell_lines:
                print(f"Processing Source data for {cell_line}")
                train_pairs_cell_line = extract_cell_line_data(cell_line, "pre_train", network, num, "sample1", ratio=None, target_cell_line=target_cell_line)
                random.shuffle(train_pairs_cell_line)  
                Source_cell_line_queues[cell_line] = train_pairs_cell_line

            print("Pre-training on Source cell lines...")
            for epoch in range(5):
                model.train()
                train_results = loop_dataset(Source_cell_line_queues, 
                                            model, 
                                            cell_line_embeddings, 
                                            optimizer, 
                                            bsize=64, 
                                            mode='train')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    else:
        print("######################")
        print("Without pretrain")
        print("######################")

    print(f"Processing Target data for {target_cell_line}")
    train_pairs = extract_cell_line_data(target_cell_line, "train", network, num, sample)
    val_pairs = extract_cell_line_data(target_cell_line, "Validation", network, num, sample)
    test_pairs = extract_cell_line_data(target_cell_line, "test", network, num, sample)
    random.shuffle(train_pairs)
    print(f"Train on Target cell line: {target_cell_line}")

    if pretrain:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Reloaded pretrain weights for fewshot_sample {sample}")
    else:
        model = TransGRN().cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        print("######################")
        print("Without pretrain")
        print("######################")

    best_auc = 0
    for epoch in range(20):
        model.train()
        ###################### training ######################
        train_results = loop_dataset(
            {f'{target_cell_line}': train_pairs}, 
            model, 
            cell_line_embeddings, 
            optimizer, bsize=64, 
            mode='train')
        print(f"\033[93mTrain Target cell line results: epoch={epoch} loss={train_results[0]:.3f}, auc={train_results[2]:.3f},  ap={train_results[3]:.3f}\033[0m")  # 黄色
        ###################### validation ######################
        model.eval()
        val_results = loop_dataset(
            {f'{target_cell_line}': val_pairs}, 
            model, 
            cell_line_embeddings,
            mode='test')
        print(f"\033[92mValidation Target cell line results: epoch={epoch} loss={val_results[0]:.3f}, auc={val_results[2]:.3f}, ap={val_results[3]:.3f}\033[0m")  # 绿色
        if val_results[2] > best_auc:
            best_auc = val_results[2]
            best_epoch = epoch
            ###################### test ######################
            model.eval()
            test_results = loop_dataset(
                {f'{target_cell_line}': test_pairs}, 
                model,
                cell_line_embeddings,
                optimizer=None, 
                bsize=64,
                mode='test')
            print(f"\033[94mTest Target cell line results for {target_cell_line}: loss={test_results[0]:.3f}, auc={test_results[2]:.3f}, ap={test_results[3]:.3f}\033[0m")  # 蓝色

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file = f'result_benchmark/{network}.txt'

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'a') as f:
        f.write(f'[{current_time}]: ')
        f.write(f"AUC={test_results[2]:.5f} AP={test_results[3]:.5f} ")
        f.write(f"{target_cell_line} {network} {num} {sample} {best_epoch}\n")


if __name__ == "__main__":
    main()