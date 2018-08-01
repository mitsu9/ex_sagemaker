from sagemaker.tensorflow import TensorFlow

def run():
    # sagemakerを使うためのインスタンス生成
    # どのparamがrequiredでどれがoptionalかはまだ調べてないのでよくわかってない
    # 一旦公式docsにあったやつを利用してる
    tf_estimator = TensorFlow(
        entry_point='tf-keras-train.py', # モデルなどについて書かれているファイル
        role='SageMakerRole', # IAMロール. SageMakerが使える権限を与えておく必要がある
        training_steps=20, # 学習時のパラメータ
        evaluation_steps=10, # 学習時のパラメータ
        train_instance_count=1, # 学習に使うインスタンスの個数. 複数指定して分散学習とかできる
        train_instance_type='ml.m5.large' # インスタンスタイプ
        )

    # 学習を開始する
    tf_estimator.fit('s3://sagemaker-ap-northeast-1-192494425048') # s3のバケット名を与える

    # 学習したモデルをデプロイする
    tf_predictor = tf_estimator.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

if __name__ == '__main__':
    run()
