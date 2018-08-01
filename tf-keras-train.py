def keras_model_fn(hyperparameters):
    # Logic to do the following:
    # 1. Instantiate the Keras model
    # 2. Compile the Keras model
    return compiled_model

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
