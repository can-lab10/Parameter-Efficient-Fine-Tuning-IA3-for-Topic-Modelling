# Topic Modeling with IA3 fine-tuning Language Models 

In this readme file you will find instructions for setting up the repository and executing the code.

## Project Overview
This project aims to fine-tune Language Models, to perform topic modeling on a collection of text paragraphs. The objective is to allow the fine-tuned models to dynamically generate meaningful topics for given texts.

## Installation
1. Clone the repository using git.
2. Build the Docker image and run the container.

Python 3.9.7 is used for the experiments.

## File Descriptions

data: Stores training and evaluation data.
eval_results: Contains evaluation results.
train_results: Contains training results.
utils.py: Contains utility functions.
train.py: Script to IA3 fine-tune the LLM on the training data.
eval.py: Script to evaluate the fine-tuned LLM on the evaluation data.
requirements.txt: Lists the Python dependencies required for the project.
Dockerfile: Defines the Docker image setup for the project.


## Running the training and evaluation

This section explains how to run the training and evaluation scripts to fine-tune the model and evaluate its performance. The training script fine-tunes the model on the provided dataset, while the evaluation script assesses the model's performance by comparing its generated topics with the ground truth.

### Fine-tune an LLM using IA3

The train.py script is used to fine-tune the specified LLM on the provided training data using IA3 (Insert Adapter Activation Algorithm). The fine-tuning process involves adjusting the model parameters to better fit the data, enabling the model to generate meaningful topics based on the input text.

```shell
  python train.py \
  --model_name google-t5/t5-small \
  --data_path data/training_data.csv \
  --output_dir train_results \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 3e-3 \ 
  --max_input_length 1024 \
  --max_target_length 128
```
--model_name: The name of the model to be fine-tuned.
--data_path: The path to the training data CSV file.
--output_dir: The directory to save the fine-tuned model.
--epochs: The number of training epochs.
--batch_size: The batch size for training.
--learning_rate: The learning rate for training.
--max_input_length: The maximum length of input sequences.
--max_target_length: The maximum length of target sequences.


### Evaluating fine-tuned LLMs

The eval.py script is used to evaluate the fine-tuned model on a separate evaluation dataset. It generates topics for the input texts and compares them with the ground truth topics using Jaccard Similarity. The evaluation results, including the generated topics by the model and their Jaccard Similarity scores, are saved in an Excel file.

```shell
  python eval.py \
  --model_name google-t5/t5-small \
  --adapter_dir training_results/adapter_checkpoints_20240615_0006 \
  --data_path data/eval_test.csv \
  --output_path eval_results/evaluation_results.xlsx
```

--model_name: The name of the model to be evaluated.
--adapter_dir: The directory of the adapter checkpoints from the fine-tuning process.
--data_path: The path to the evaluation data CSV file.
--output_path: The path to save the evaluation results as an Excel file.
--max_input_length: The maximum length of input sequences.
--max_target_length: The maximum length of target sequences.