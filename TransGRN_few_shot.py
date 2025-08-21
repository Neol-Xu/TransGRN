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
    finetune = args.finetune
    target_cell_line = args.target_cell_line
    source_cell_lines = args.source_cell_lines.split(",")

    checkpoint_path = f'check_point/{network}/{target_cell_line}_{num}/sample1/model_weights.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model = TransGRN().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load cell line embeddings
    cell_line_embeddings = load_cell_line_embeddings()
    
    ###################### pre-training ######################
    print("######################")
    print("With pre-train")
    print("######################")
    if os.path.exists(checkpoint_path):
        print(f"Loaded checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            print(f"\033[92mPre-train Source cell lines results: epoch={epoch} loss={train_results[0]:.3f}, auc={train_results[2]:.3f},  ap={train_results[3]:.3f}\033[0m")  # 绿色
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    ###################### fine-tuning ######################
    shots = ['5%']
    for shot in shots:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Reloaded pretrain weights for fewshot_sample {sample}")

        model.train()
        if finetune:
            print(f"Processing Target data for {target_cell_line}")
            few_shot_pairs = extract_cell_line_data(target_cell_line, "few_shot_train", network, num, sample, shot)
            random.shuffle(few_shot_pairs)
            print(f"Few-shot fine-tuning on Target cell line: {target_cell_line}")
            for epoch in range(5):
                few_shot_results = loop_dataset({f'{target_cell_line}': few_shot_pairs}, 
                                                model, 
                                                cell_line_embeddings, 
                                                optimizer, 
                                                bsize=64, 
                                                mode='train')
                print(f"\033[93mFew-shot Target cell line results: epoch={epoch} loss={few_shot_results[0]:.3f}, auc={few_shot_results[2]:.3f},  ap={few_shot_results[3]:.3f}\033[0m")  
        ###################### test ######################
        test_pairs = extract_cell_line_data(target_cell_line, "few_shot_test", network, num, sample, shot)
        model.eval()
        test_results = loop_dataset(
            {f'{target_cell_line}': test_pairs}, 
            model,
            cell_line_embeddings,
            optimizer=None, 
            bsize=64,
            mode='test')
        print(f"\033[94mTest Target cell line results for {target_cell_line}: loss={test_results[0]:.3f}, auc={test_results[2]:.3f}, ap={test_results[3]:.3f}\033[0m")  

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_file = f'result_fewshot/{network}/{shot}.txt'
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'a') as f:
            f.write(f'[{current_time}]: ')
            f.write(f"AUC={test_results[2]:.5f} AP={test_results[3]:.5f} {network} {num} fewshot_sample {sample} Source_sample {sample} Final test for {target_cell_line}\n")

if __name__ == "__main__":
    main()  