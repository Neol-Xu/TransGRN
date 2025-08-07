import csv

gpt_embeddings_file = 'mouse_GPT_3_5_gene_embeddings.csv'
gpt_embeddings = {}
with open(gpt_embeddings_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    gpt_header = next(csv_reader) 
    for row in csv_reader:
        gene_name = row[0]
        embedding = list(map(float, row[1:]))
        gpt_embeddings[gene_name] = embedding

# load BL--ExpressionData data
expression_data_file = 'BL--ExpressionData.csv'
expression_data = []
with open(expression_data_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    expression_header = next(csv_reader)
    for row in csv_reader:
        expression_data.append(row)

GPT_DIM = len(next(iter(gpt_embeddings.values())))

# Align embeddings and reorder according to the gene names in BL--ExpressionData
aligned_embeddings = []
for row in expression_data:
    gene_name = row[0]
    if gene_name in gpt_embeddings:
        aligned_embeddings.append([gene_name] + gpt_embeddings[gene_name])
    else:
        # If the corresponding gene name is not found in the GPT embeddings, fill with a zero vector
        aligned_embeddings.append([gene_name] + [0.0] * GPT_DIM)

aligned_embeddings_file = 'Aligned_GPT_3_5_gene_embeddings.csv'
with open(aligned_embeddings_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Gene"] + [f"Dim_{i}" for i in range(GPT_DIM)])
    csv_writer.writerows(aligned_embeddings)

print(f"The aligned embeddings have been saved as {aligned_embeddings_file}")


