# TransGRN: a transfer learning-based method for inferring gene regulatory networks across cell lines
![image](https://github.com/Neol-Xu/TransGRN/blob/main/figure/Flowchat.png)
# Usage
## 1. Set Up Conda Environment
Before running any scripts, set up the required environment using the provided environment.yml file. Run the following command to create the environment:
```shell
conda env create -f environment.yml
```
## 2. Data generation
Execute the Python files in the `data_generate` folder in sequence to obtain gene description embeddings. Subsequently, run the following commands in order to split the dataset.
```shell
python data_split/benchmark_data.py
python data_split/Few_shot_data.py
python data_split/Source_data.py
python data_split/cold_start_data.py
python data_split/cold_start_source_data.py
```
## Gene regulatory networks inference
Run the model to infer gene regulatory networks by following commands.
```shell
bash script\benchmark.sh
bash script\Fewshot.sh
```
# Contact
For any inquiries, feel free to raise issues or contact me via email at xuge88437@gmail.com
