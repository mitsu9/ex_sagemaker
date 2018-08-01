from sagemaker.tensorflow import TensorFlow

def run():
    tf_estimator = TensorFlow(entry_point='tf-keras-train.py', role='SageMakerRole',
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge')
    tf_estimator.fit('s3://bucket/path/to/training/data')

    tf_predictor = tf_estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

if __name__ == '__main__':
    run()
