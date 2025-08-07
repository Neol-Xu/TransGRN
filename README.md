# TransGRN: a transfer learning-based method for inferring gene regulatory networks across cell lines
![image](https://github.com/Neol-Xu/TransGRN/blob/master/figure/Flowchat.png)
# Dependencies Used in This Project
- Python  3.8.0
- PyTorch 1.12.1
- cuda 11.3
# Usage
## 1. Generate gene description embeddings  
Run the python scripts in the `data_generate` folder sequentially to collect the required gene function description summaries from the NCBI database, and then use the GPT-3.5 model API to convert the collected text information into gene description embeddings.
## 2. Generate network data sets
Run the Python scripts in the `data_split` folder for data splitting sequentially:
```python
python data_split/Source_data.py
python data_split/benchmark_data.py
python data_split/Few_shot_data.py
python data_split/cold_cold_start_source_data.py
python data_split/cold_start_data.py
```
## 3. Run Inference
Setup the required environment using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Then we can run the model:
```bash
bash script/benchmark.sh
```
# Contact
For any inquiries, feel free to raise issues or contact me via email at xuge88437@gmail.com
.
