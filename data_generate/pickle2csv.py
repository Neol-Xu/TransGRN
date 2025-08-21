import pickle
import csv
import numpy as np

pickle_file_path = "mouse_GPT_3_5_gene_embeddings.pickle"
with open(pickle_file_path, "rb") as fp:
    gpt_gene_name_to_embedding_clean_text = pickle.load(fp)

GPT_DIM = len(next(iter(gpt_gene_name_to_embedding_clean_text.values())))

csv_file_path = "mouse_GPT_3_5_gene_embeddings.csv"
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Gene"] + [f"Dim_{i}" for i in range(GPT_DIM)])
    for gene, embedding in gpt_gene_name_to_embedding_clean_text.items():
        csv_writer.writerow([gene] + embedding.tolist())

print(f"The embedding results have been saved as {csv_file_path}")