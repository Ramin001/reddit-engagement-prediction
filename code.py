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

from pprint import pprint


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

# split data: 10% test. the rest is randomly split 80-20 between train and validation
split_dataset_into_train_test(raw_path, test_ratio=0.1, val_ratio=0.2)


###################################################
# import data                                     #
###################################################
# function to create a keras dataset generator for cleaning reddit data
def reddit_post_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            
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
    'created': tf.TensorSpec(shape=(), dtype=tf.float64),
    'created_utc': tf.TensorSpec(shape=(), dtype=tf.float64),
    'retrieved_on': tf.TensorSpec(shape=(), dtype=tf.float64)
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



