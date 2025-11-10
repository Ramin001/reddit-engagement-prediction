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

# Define the file path
file_path = "wallstreetbets_submissions"

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
dataset = tf.data.Dataset.from_generator(
    lambda: reddit_post_generator(file_path),
    output_signature=output_signature
)

# Configure the dataset for GPU performance
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128 

dataset = dataset.cache()  # Cache the data to memory
dataset = dataset.batch(BATCH_SIZE)  # Create batches
dataset = dataset.prefetch(AUTOTUNE)  # Prefetch next batch while GPU is processing



# Look at individual entries directly from the generator to see progress
print("Reading posts...")
count = 0
for entry in reddit_post_generator(file_path):
    if count >= 5:  # Only look at first 5 entries
        break
        
    count += 1
    print(f"\nEntry #{count}:")
    print("Title:", entry['title'])
    print("Text:", entry['selftext'][:100] + "..." if len(entry['selftext']) > 100 else "")
    
    # Check numeric fields
    numeric_fields = ['score', 'num_comments', 'ups', 'downs', 'num_reports']
    print("\nNumeric values:")
    for field in numeric_fields:
        value = entry[field]
        print(f"{field}: {'NaN' if np.isnan(value) else value}")
    
    # Check timestamp fields
    timestamp_fields = ['created', 'created_utc', 'retrieved_on']
    print("\nTimestamp values:")
    for field in timestamp_fields:
        value = entry[field]
        if np.isnan(value):
            print(f"{field}: NaN")
        else:
            print(f"{field}: {pd.to_datetime(value, unit='s')}")
    
    print("-" * 80)  
    

print("Dataset ready for GPU processing with batch size", BATCH_SIZE)


################################################### 
### split dataset into training, validation, and test sets
#####################################################
total_size = 0
