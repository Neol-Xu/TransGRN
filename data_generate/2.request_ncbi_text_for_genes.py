import requests
from bs4 import BeautifulSoup
import html2text
import mygene
import json
import pandas as pd
import os
import time

def rough_text_from_gene_name(gene_number, retries=5, timeout=30):
    """
    Retrieve gene summary text from NCBI.

    Parameters:
        gene_number (str): NCBI ID of the gene.
        retries (int): Maximum number of retries.
        timeout (int): Request timeout in seconds.

    Returns:
        summary_text (str): Gene summary text.
    """
    url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_number}"
    summary_text = ''
    soup = None

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status() 

            soup = BeautifulSoup(response.content, 'html.parser')

            summary_tab = soup.find('div', {'class': 'rprt-section gene-summary'})

            if summary_tab:
                html_to_text = html2text.HTML2Text()
                html_to_text.ignore_links = True  

                summary_text = html_to_text.handle(str(summary_tab))

                parts_to_remove = [
                    "##  Summary\n",
                    "NEW",
                    'Try the newGene table',
                    'Try the newTranscript table',
                    '**',
                    "\nGo to the top of the page Help\n"
                ]
                for part in parts_to_remove:
                    summary_text = summary_text.replace(part, ' ')

                summary_text = summary_text.replace('\n', ' ')

                summary_text = ' '.join(summary_text.split())
            else:
                print(f"Summary section for gene {gene_number} not found.")

            break

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f'Request timed out or connection error, retrying {attempt + 1}/{retries}...')
            time.sleep(5) 
        except requests.exceptions.HTTPError as e:
            print(f'HTTP error: {e}')
            break
        except requests.exceptions.ChunkedEncodingError as e:
            print(f'Chunked encoding error: {e}')
            time.sleep(5) 
        except Exception as e:
            print(f'Unknown error: {e}')
            break
    else:
        print(f'Request failed, maximum retry limit of {retries} reached.')

    return summary_text

def save_summary_to_json(summary_dict, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        existing_data.update(summary_dict)
    else:
        existing_data = summary_dict

    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    print(f"Gene summary information has been saved to {output_file}")

mg = mygene.MyGeneInfo()

mouse_genes_df = pd.read_csv('mouse_gene_summary.csv')
mouse_genes = mouse_genes_df['Gene Name'].tolist()

# Crawl and save gene summary information
species = 'mouse'
gene_name_to_tax_id = {}
missing_genes = []
for gene_name in mouse_genes:
    print('Querying gene:', gene_name)
    result = mg.query(gene_name, scopes='symbol', species=species)
    if result and 'hits' in result and len(result['hits']) > 0:
        gene_name_to_tax_id[gene_name] = result['hits'][0]['_id']
    else:
        missing_genes.append(gene_name)

# gene_name_to_tax_id to JSON
gene_id_file = 'mouse_gene_name_to_tax_id.json'
with open(gene_id_file, 'w') as f:
    json.dump(gene_name_to_tax_id, f, indent=4, ensure_ascii=False)
print(f"Gene name to Gene ID mapping has been saved to {gene_id_file}")

with open('mouse_gene_name_to_tax_id.json', 'r') as f:
    gene_name_to_tax_id = json.load(f)
output_file = 'mouse_gene_summary.json'
for gene_name in mouse_genes:
    if gene_name in gene_name_to_tax_id:
        print('Fetching gene:', gene_name)
        page_id = gene_name_to_tax_id[gene_name]
        parsed_text = rough_text_from_gene_name(page_id)
        save_summary_to_json({gene_name: parsed_text}, output_file)