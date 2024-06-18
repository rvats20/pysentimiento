#!/bin/bash

# Define the models
models=("tweetnlp" "pysentimiento")

# Check if not --lang is passed
if [ -z "$1" ]
then
    langs=("en" "es" "it")
else
    langs=($1)
fi

if [ -z "$2" ]
then
    tasks=("sentiment" "hate_speech")
else
    tasks=($2)
fi

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