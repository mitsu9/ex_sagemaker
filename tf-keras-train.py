import tensorflow as tf
# from tf.keras.models import Sequential
# from tf.keras.layers import Embedding, Flatten, Dense, Activation
import logging

def keras_model_fn(hyperparameters):
    logging.info('In keras_model_fn: hyperparameters are shown below...')
    logging.info(hyperparameters)
    # parameters
    vocab_size = 8000
    embedding_dim = 256
    input_length = # 入力データ(行列)の列数
    last_layer_dim = 1
    # Logic to do the following:
    # 1. Instantiate the Keras model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length, embeddings_initializer='he_normal'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(last_layer_dim))
    model.add(tf.keras.layers.Activation('sigmoid'))
    # 2. Compile the Keras model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def train_input_fn(training_dir, hyperparameters):
  # Logic to the following:
  # 1. Reads the **training** dataset files located in training_dir
  # 2. Preprocess the dataset
  # 3. Return 1)  a dict of feature names to Tensors with
  # the corresponding feature data, and 2) a Tensor containing labels
  return features, labels

def eval_input_fn(training_dir, hyperparameters):
  # Logic to the following:
  # 1. Reads the **evaluation** dataset files located in training_dir
  # 2. Preprocess the dataset
  # 3. Return 1)  a dict of feature names to Tensors with
  # the corresponding feature data, and 2) a Tensor containing labels
  return features, labels
