#!/bin/bash
dataset_types=("Specific")
tf_counts=("1000")
samples=("sample1")

source_celltypes=("hHEP" "mDC" "mESC" "mHSC-E" "mHSC-GM" "mHSC-L")
target_celltype="hESC"

for dataset_type in "${dataset_types[@]}"; do
    for tf_count in "${tf_counts[@]}"; do
        for sample in "${samples[@]}"; do
            source_celltypes_str=$(IFS=,; echo "${source_celltypes[*]}")

            echo "Running with:"
            echo "  Network Type: $dataset_type"
            echo "  TF Count: $tf_count"
            echo "  Sample: $sample"
            echo "  Target Celltype: $target_celltype"
            echo "  Source Celltypes: $source_celltypes_str"

            # export CUDA_VISIBLE_DEVICES='0'
            python TransGRN_benchmark.py \
                --network "$dataset_type" \
                --num "$tf_count" \
                --sample "$sample" \
                --source_cell_lines "$source_celltypes_str" \
                --target_cell_line "$target_celltype" 
        done
    done
done
