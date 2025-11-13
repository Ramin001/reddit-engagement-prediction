# reddit file
# Import Required Libraries
import os
import json
import html

import pandas as pd
import keras as ks
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers 
import tensorflow.keras.losses as losses

import matplotlib.pyplot as plt     
from pprint import pprint
# Disable XLA devices to avoid GPU memory issues
#os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
###################################################
# create datasets: training, validation, and test #
###################################################
raw_path = "wallstreetbets_submissions"
# data is chronological, so take test from the end
def split_dataset_into_train_test(file_path, test_ratio=0.1, val_ratio=0.2):
    import random
    total_lines = sum(1 for _ in open(file_path))
    nsplit = int(total_lines*(1 - test_ratio))
    outs = [open('train_ds', 'w', encoding='utf-8'), 
            open('valid_ds', 'w', encoding='utf-8'),
            open('test_ds', 'w', encoding='utf-8')]
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = random.random()
            if   i < nsplit and r >  0.2: outs[0].write(line)
            elif i < nsplit and r <= 0.2: outs[1].write(line)
            else: outs[2].write(line)
    for out in outs: out.close()

first_run = not (os.path.exists('train_ds') and os.path.exists('valid_ds') and os.path.exists('test_ds'))
# split data: 10% test. the rest is randomly split 80-20 between train and validation
if first_run:
    split_dataset_into_train_test(raw_path, test_ratio=0.1, val_ratio=0.2)


###################################################
# import data                                     #
###################################################
# function to create a keras dataset generator for cleaning reddit data
def reddit_post_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            
            # filter out unwanted columns
            entry = {k: entry.get(k, float('nan')) for k in ['title', 'selftext', 'score', 'num_comments', 
                                           'ups', 'downs', 'num_reports',
                                           'created', 'created_utc', 'retrieved_on']}
            
            # clean text columns: fix html elements, lower case, and strip whitespace
            for column in ['title', 'selftext']:
                if column in entry and isinstance(entry[column], str):
                    entry[column] = html.unescape(entry[column])
                    entry[column] = entry[column].strip().lower()
                else:
                    entry[column] = ""

                    
            # clean date columns: convert to timestamps with NaN for missing values
            for column in ['created', 'created_utc', 'retrieved_on']:
                if column in entry and entry[column]:
                    try:
                        entry[column] = float(entry[column])  # Convert to float to support NaN
                    except (ValueError, TypeError):
                        entry[column] = np.nan
                else:
                    entry[column] = np.nan

            
            # clean numeric columns: convert to float types to support NaN
            for column in ['score', 'num_comments', 'ups', 'downs', 'num_reports']:
                if column in entry:
                    try:
                        entry[column] = float(pd.to_numeric(entry[column], errors='coerce'))
                    except:
                        entry[column] = np.nan
                else:
                    entry[column] = np.nan

            yield entry

# Define the output signature for the dataset
output_signature = {
    'title': tf.TensorSpec(shape=(), dtype=tf.string),
    'selftext': tf.TensorSpec(shape=(), dtype=tf.string),
    'score': tf.TensorSpec(shape=(), dtype=tf.float32),
    'num_comments': tf.TensorSpec(shape=(), dtype=tf.float32),
    'ups': tf.TensorSpec(shape=(), dtype=tf.float32),
    'downs': tf.TensorSpec(shape=(), dtype=tf.float32),
    'num_reports': tf.TensorSpec(shape=(), dtype=tf.float32),
    'created': tf.TensorSpec(shape=(), dtype=tf.float32),
    'created_utc': tf.TensorSpec(shape=(), dtype=tf.float32),
    'retrieved_on': tf.TensorSpec(shape=(), dtype=tf.float32)
}

# Create the dataset with GPU optimization
train_ds = tf.data.Dataset.from_generator(
    lambda: reddit_post_generator('train_ds'),
    output_signature=output_signature
)
valid_ds = tf.data.Dataset.from_generator(
    lambda: reddit_post_generator('valid_ds'),
    output_signature=output_signature
)
test_ds = tf.data.Dataset.from_generator(
    lambda: reddit_post_generator('test_ds'),
    output_signature=output_signature
)
# Configure the dataset for GPU performance
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128

train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE) 
valid_ds = valid_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE) 
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Dataset ready for GPU processing with batch size", BATCH_SIZE)


###################################################
# train the model                                 #
###################################################

# Define text vectorization layers
max_features = 20000    
sequence_length = 200
title_vectorizer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
selftext_vectorizer = layers.TextVectorization(
    max_tokens=max_features,    
    output_mode='int',
    output_sequence_length=sequence_length  
)   
# Adapt vectorizers to the training data
title_texts = train_ds.map(lambda x: x['title'])    
selftext_texts = train_ds.map(lambda x: x['selftext'])
title_vectorizer.adapt(title_texts)    
selftext_vectorizer.adapt(selftext_texts)   

#print vocabulary samples
print("Title vocabulary sample:", title_vectorizer.get_vocabulary()[:20])
print("Selftext vocabulary sample:", selftext_vectorizer.get_vocabulary()[:20])     

# Define the model architecture
def build_model(Embedding_dim=64):
    title_input = layers.Input(shape=(1,), dtype=tf.string, name='title_input') 
    title_vectorized = title_vectorizer(title_input)
    title_embedded = layers.Embedding(input_dim=max_features, output_dim=Embedding_dim, name='title_embedding')(title_vectorized)

    selftext_input = layers.Input(shape=(1,), dtype=tf.string, name='selftext_input') 
    selftext_vectorized = selftext_vectorizer(selftext_input)   
    selftext_embedded = layers.Embedding(input_dim=max_features, output_dim=Embedding_dim, name='selftext_embedding')(selftext_vectorized)
    
    # Combine text features
    combined = layers.Concatenate(name='concat')([title_embedded, selftext_embedded])
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(combined)
    x = layers.Dense(Embedding_dim, activation='relu', name='dense_1')(x)
    output = layers.Dense(1, activation='linear', name='output')(x)
    model = ks.Model(inputs=[title_input, selftext_input], outputs=output)
    return model

# Build the model
model = build_model(Embedding_dim=64)

# Compile the model
model.compile(
    optimizer='adam',
    loss=losses.MeanSquaredError(),
    metrics=['mse']
)
summary = model.summary()
pprint(summary)

###################################################
# train the model                                 #
###################################################
with tf.device('/CPU:0'):
    history = model.fit(
        train_ds.map(lambda x: ({"title_input": x['title'], "selftext_input": x['selftext']}, x['score'])),
        validation_data=valid_ds.map(lambda x: ({"title_input": x['title'], "selftext_input": x['selftext']}, x['score'])),
        epochs=30
    )
###################################################
# evaluate the model                              #     
###################################################
eval_results = model.evaluate(
    test_ds.map(lambda x: ({"title_input": x['title'], "selftext_input": x['selftext']}, x['score']))
)
pprint(eval_results)

# plot training history

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')     
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()



