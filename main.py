import argparse
import os
from tensorflow import keras
import losses, train, GhostFaceNets
import tensorflow as tf

from pathlib import Path


def main(params):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.mixed_precision.set_global_policy("mixed_float16")

    # PetFace dataset
    if not params.data_path.endswith('.csv'):
        raise ValueError(f'Provided data path {params.data_path} does not end with .csv. Please provide a valid data path.')
    if not os.path.exists(params.data_path):
        raise FileNotFoundError(f'Provided data path {params.data_path} does not exist. Please provide a valid data path.')
    data_path = params.data_path

    # Create Work dir
    Path(params.work_dir).mkdir(parents=True, exist_ok=True)

    # GhostFaceNetV1 Strides 1
    # basic_model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC',
    #                                            bn_momentum=0.9, bn_epsilon=1e-5, scale=True, use_bias=True, strides=1)
    # basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
    # basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)

    if params.loss == 'arcface':
        save_path = os.path.join(params.work_dir, 'ghostV1-1.3-1-(A).h5')
        basic_model = 'weights/GhostFaceNetV1-1.3-1-ArcFace.h5'
    else:
        save_path = os.path.join(params.work_dir, 'ghostV1-1.3-1-(C).h5')
        basic_model = 'weights/GhostFaceNetV1-1.3-1-CosFace.h5'

    tt = train.Train(data_path,
                     save_path=save_path,
                     basic_model=basic_model, model=None, lr_base=0.1, lr_decay=0.5, lr_decay_steps=45, lr_min=1e-5,
                     batch_size=128, random_status=0, eval_freq=1, output_weight_decay=1)

    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    if params.loss == 'arcface':
        sch = [
            {"loss": losses.ArcfaceLoss(scale=32), "epoch": 1, "optimizer": optimizer},
            {"loss": losses.ArcfaceLoss(scale=64), "epoch": 50},
        ]
    else:
        sch = [
            {"loss": losses.CosFaceLoss(scale=32), "epoch": 1, "optimizer": optimizer},
            {"loss": losses.CosFaceLoss(scale=64), "epoch": 50},
        ]
    tt.train(sch, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GhostFaceNet')
    parser.add_argument('--loss', type=str, default='arcface', help='Loss function: arcface or cosface')
    parser.add_argument('--data', type=str, dest='data_path', required=True, help='Path to the train.csv')
    parser.add_argument('--output', type=str, dest='work_dir', default='work_dir', help='Path to the working directory.')
    args = parser.parse_args()
    main(args)
