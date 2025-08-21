import json
import numpy as np
import pickle
from openai import OpenAI

API_KEY = ''  # Replace with your own API key
BASE_URL = ''  # Modify to the base_url you want to use

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def get_gpt_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ") 
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float" 
        )
        embedding = response.data[0].embedding 
        return np.array(embedding)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# load gene summary data from a JSON file
with open(r"mouse_gene_summary", 'r') as file:
    NCBI_cleaned_summary_of_genes = json.load(file)

gpt_gene_name_to_embedding_clean_text = {}

GPT_DIM = 1536 

for key, text in sorted(NCBI_cleaned_summary_of_genes.items()):
    print(f'Processing gene: {key}')
    if text == '':
        gpt_gene_name_to_embedding_clean_text[key] = np.zeros(GPT_DIM)
    else:
        embedding = get_gpt_embedding(text)
        if embedding is not None:
            gpt_gene_name_to_embedding_clean_text[key] = embedding
        else:
            print(f"Skipping gene {key} due to API error, embedding vector could not be retrieved.")

# save the embeddings to a pickle file
with open("mouse_GPT_3_5_gene_embeddings.pickle", "wb") as fp:
    pickle.dump(gpt_gene_name_to_embedding_clean_text, fp)