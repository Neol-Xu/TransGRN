import pandas as pd
import os

def create_fewshot_sets():
    data_types = ["hESC", "hHEP", "mDC", "mHSC-E", "mESC", "mHSC-GM", "mHSC-L"]
    net_types = ["Specific", "Non-Specific", "STRING"]
    nums = [500, 1000]
    
    ratio = 0.05  
    base_path = os.getcwd()

    for net_type in net_types:
        for num in nums:
            for data_type in data_types:
                for sample_num in range(1, 6):

                    print(f"{net_type} {num} {data_type} {sample_num} | Ratio: {ratio}")

                    train_path = os.path.join(
                        base_path,
                        "benchmark",
                        net_type,
                        f"{data_type}_{num}",
                        f"sample{sample_num}",
                        "Train_set.csv"
                    )
                    
                    if not os.path.exists(train_path):
                        print(f"File not found: {train_path}")
                        continue
                    
                    train_df = pd.read_csv(train_path)
                    pos_samples = train_df[train_df['Label'] == 1]
                    neg_samples = train_df[train_df['Label'] == 0]

                    n_pos = max(10, int(len(pos_samples) * ratio))
                    n_neg = min(len(neg_samples), n_pos)

                    fewshot_pos = pos_samples.sample(n=n_pos, random_state=42)
                    fewshot_neg = neg_samples.sample(n=n_neg, random_state=42)
                    
                    fewshot_df = pd.concat([fewshot_pos, fewshot_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

                    fewshot_dir = os.path.join(
                        base_path,
                        "Fewshot",
                        f"{int(ratio*100)}%", 
                        net_type,
                        f"{data_type}_{num}",
                        f"sample{sample_num}"
                    )
                    os.makedirs(fewshot_dir, exist_ok=True)
                    
                    fewshot_path = os.path.join(fewshot_dir, "Train_set.csv")
                    fewshot_df.to_csv(fewshot_path, index=False)

                    original_test_path = os.path.join(
                        base_path,
                        "benchmark",
                        net_type,
                        f"{data_type}_{num}",
                        f"sample{sample_num}",
                        "Test_set.csv"
                    )
                    test_df = pd.read_csv(original_test_path)
                    fewshot_test_path = os.path.join(fewshot_dir, "Test_set.csv")
                    test_df.to_csv(fewshot_test_path, index=False)
                    
                    print(f"Created few-shot set at: {fewshot_path}")

if __name__ == "__main__":
    create_fewshot_sets()
