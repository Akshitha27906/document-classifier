# document-classifier
A document classifier that categorizes news articles

A machine learning model that reads any news sentence 
and automatically classifies it into a category.

## Categories
- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Science/Tech

## Accuracy
91.75% on 7,600 test articles

## Dataset
AG News - 120,000 news articles from HuggingFace

## How it works
1. Downloads 120,000 labeled news articles
2. Cleans and lowercases the text
3. Converts words into numbers using TF-IDF
4. Trains a Logistic Regression model
5. User can type any sentence and get the category

## How to run
Install libraries:
pip install scikit-learn datasets numpy

Then run:
python train.py

## Example
Type: Virat Kohli scores century
Output: Category: Sports
