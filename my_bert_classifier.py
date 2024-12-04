# Thomas Roberts
# CS 4742 Natural Language Processing
# Professor Alexiou
# November 27, 2024

from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

class MyBert:
    def __init__(self, ds_train, ds_test, lr=2e-5):
        self.ds_train = tf.data.Dataset.from_tensor_slices(dict(ds_train))
        self.ds_test = tf.data.Dataset.from_tensor_slices(dict(ds_test))

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Initialize BERT model
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Initialize training variables
        self.max_length = 512
        return None
    # end __init__()

    def preprocess_ds(self, batch_size):
        ds_train_encoded = self.encode_examples(self.ds_train).shuffle(10000).batch(batch_size)
        ds_test_encoded = self.encode_examples(self.ds_test).batch(batch_size)
        return ds_train_encoded, ds_test_encoded
    # end preprocess()

    def train(self, ds_train_encoded, ds_test_encoded, epoch):
        """
        Training function

        Parameters
        ----------
        ds_train_encoded
        ds_test_encoded
        epoch

        Returns
        --------
        """
        # Train model
        bert_history = self.model.fit(ds_train_encoded, epochs=epoch, validation_data=ds_test_encoded)

        # Example of testing
        test_sentence = "This is a really good movie. I loved it and will watch again"
        predict_input = self.tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")

        tf_output = self.model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)

        labels = ['Negative', 'Positive']
        label = tf.argmax(tf_prediction, axis=1).numpy()
        print(labels[label[0]])
        return bert_history
    # end train()

    def convert_example_to_feature(self, review):
        return self.tokenizer.encode_plus(review,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          padding='max_length',
                                          return_attention_mask=True)
    # end convert_example_to_feature()

    def map_example_to_dict(self, input_ids, attention_masks, token_type_ids, labels):
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        }, labels
    # end map_example_to_dict()

    def encode_examples(self, ds, limit=-1):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        if limit > 0:
            ds = ds.take(limit)

        for example in tfds.as_numpy(ds):
            review = example['text']
            label = example['label']
        
            bert_input = self.convert_example_to_feature(review.decode())
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append(label)

        return tf.data.Dataset.from_tensor_slices(({
            'input_ids': input_ids_list,
            'token_type_ids': token_type_ids_list,
            'attention_mask': attention_mask_list
        }, label_list)).map(lambda x, y: self.map_example_to_dict(x['input_ids'], x['attention_mask'], x['token_type_ids'], y))
    # end encode_examples()

    def plot_accuracy(self, history):
        """
        Plots the accuracy metric from the training history of a TensorFlow model.
        
        Parameters:
        history (tf.keras.callbacks.History): The history object returned from model.fit()
        """
        # Extract accuracy values
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Get the number of epochs
        epochs = range(1, len(accuracy) + 1)

        # Plot training and validation accuracy per epoch
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
    # end plot_accuracy
# end class MyBert
