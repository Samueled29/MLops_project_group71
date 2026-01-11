# Apple Disease Detection for Danish Agriculture using Vision Transformers

This project develops a Vision Transformer (ViT)–based binary classification system that identifies whether apples are healthy or diseased from image data. We aim to achieve this by using a vision transformer model trained on the Fruit and Vegetable Disease dataset obtained from Kaggle.

## Vision Transformer

Vision Transformer (ViT) is a transformer adapted for computer vision tasks. An image is split into smaller fixed-sized patches which are treated as a sequence of tokens, similar to words for NLP tasks. ViT requires less resources to pretrain compared to convolutional architectures and its performance on large datasets can be transferred to smaller downstream tasks.
In this project, a pretrained Vision Transformer is fine-tuned for binary classification. The original classification head is replaced with a task-specific head that outputs a single logit, which is interpreted as the probability of an apple being diseased or healthy.

## Dataset

The Fruit and Vegetable Disease dataset (https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data.) contains a comprehensive collection of images for 14 different types of fruits and vegetables but for this study only apple images are used to improve focus training efficiency and relevance to Danish agricultural production.

## Training of the model:

The model has been trained and validated using the following commands
uv run python -m fruit_and_vegetable_disease.train --batch-size 4 --epochs 5 --lr 0.0001
uv run python -m fruit_and_vegetable_disease.evaluate models/model.pth --batch-size 8

