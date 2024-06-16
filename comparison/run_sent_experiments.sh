#!/bin/bash

# Define the models
models=("vader" "textblob" "stanza" "tweetnlp" "pysentimiento")
langs=("en")

for lang in "${langs[@]}"
do

    # Loop over the models
    for model in "${models[@]}"
    do
        # Define the output file name
        output_file="results/${lang}-sentiment-${model}.csv"

        # Run the python script with the current model and output file
        python predict.py --model $model --output $output_file
    done
done