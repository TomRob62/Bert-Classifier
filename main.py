# Thomas Roberts
# CS 4742 Natural Language Processing
# Professor Alexiou
# November 28, 2024

import requests
import pandas as pd
from my_bert_classifier import MyBert

def download_data(train_row = 7500, test_row = 1500): 
    """
    Downloads amazon review dataset

    Returns
    --------
    train_data: Dataframe
    test_data: Dataframe
    """
    # Download test data
    url_test = 'https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/test_amazon.csv'
    res = requests.get(url_test, allow_redirects=True)
    with open('test_amazon.csv', 'wb') as file:
        file.write(res.content)

    # Download train data
    url_train = 'https://raw.githubusercontent.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/refs/heads/master/dataset/train_amazon.csv'
    res = requests.get(url_train, allow_redirects=True)
    with open('train_amazon.csv', 'wb') as file:
        file.write(res.content)

        # Load data
    train_data = pd.read_csv('train_amazon.csv', nrows=train_row)
    test_data = pd.read_csv('test_amazon.csv', nrows=test_row)
    return train_data, test_data
# end download data

if __name__ == '__main__':
    train_data, test_data = download_data()
    bert = MyBert(train_data, test_data)
    train_ds, test_ds = bert.preprocess_ds(10)
    history = bert.train(train_ds, test_ds, epoch=1)

    