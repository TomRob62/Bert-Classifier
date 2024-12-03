import requests
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow_datasets as tfds
import tensorflow as tf

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
train_data = pd.read_csv('train_amazon.csv', nrows=7500)
test_data = pd.read_csv('test_amazon.csv', nrows=1500)

# Convert data to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices(dict(train_data))
ds_test = tf.data.Dataset.from_tensor_slices(dict(test_data))

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_length = 512
batch_size = 10

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True, 
                                 max_length=max_length, 
                                 padding='max_length', 
                                 return_attention_mask=True)

def map_example_to_dict(input_ids, attention_masks, token_type_ids, labels):
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, labels

def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if limit > 0:
        ds = ds.take(limit)

    for example in tfds.as_numpy(ds):
        review = example['text']
        label = example['label']
        
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)

    return tf.data.Dataset.from_tensor_slices(({
        'input_ids': input_ids_list,
        'token_type_ids': token_type_ids_list,
        'attention_mask': attention_mask_list
    }, label_list)).map(lambda x, y: map_example_to_dict(x['input_ids'], x['attention_mask'], x['token_type_ids'], y))

# Model and training setup
learning_rate = 2e-5
number_of_epochs = 1

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Encode datasets
ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)
ds_test_encoded = encode_examples(ds_test).batch(batch_size)

# Train model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

# Example of testing
test_sentence = "This is a really good movie. I loved it and will watch again"
predict_input = tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")

tf_output = model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1)

labels = ['Negative', 'Positive']
label = tf.argmax(tf_prediction, axis=1).numpy()
print(labels[label[0]])
