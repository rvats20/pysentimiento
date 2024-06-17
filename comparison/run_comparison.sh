#!/bin/bash

# Define the models
models=("tweetnlp" "pysentimiento")
langs=("en" "es")
tasks=("sentiment" "hate_speech")

for task in "${tasks[@]}"
do
    for lang in "${langs[@]}"
    do

        # Loop over the models
        for model in "${models[@]}"
        do
            # Define the output file name
            output_file="results/${lang}-${task}-${model}.csv"

            # Run the python script with the current model and output file
            python predict.py --model $model --output $output_file --lang $lang --task $task
        done
    done
done