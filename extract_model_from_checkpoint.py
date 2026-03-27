import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

import GhostFaceNets
import losses
import train

data_path_base = '/mnt/data/afarec/data/PetFace'
data_path = ''
work_dir = ''


def export_from_epoch(load_path, epoch):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.mixed_precision.set_global_policy("mixed_float16")

    # PetFace dataset
    if not data_path.endswith('.csv'):
        raise ValueError(f'Provided data path {data_path} does not end with .csv. Please provide a valid data path.')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Provided data path {data_path} does not exist. Please provide a valid data path.')

    # Create Work dir
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # GhostFaceNetV2 Strides 1
    basic_model = GhostFaceNets.buildin_models("ghostnetv2", dropout=0, emb_shape=512,
                                               output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
    basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
    basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)

    if loss == 'arcface':
        save_path = os.path.join(work_dir, 'ghostV2-1.3-1-(A).h5')
    else:
        save_path = os.path.join(work_dir, 'ghostV2-1.3-1-(C).h5')

    tt = train.Train(data_path,
                     save_path=save_path,
                     basic_model=basic_model, model=None, lr_base=0.1, lr_decay=0.5, lr_decay_steps=45, lr_min=1e-5,
                     batch_size=128, random_status=0, eval_freq=1, output_weight_decay=1)

    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    if loss == 'arcface':
        sch = [
            {"loss": losses.ArcfaceLoss(scale=32), "epoch": 1, "optimizer": optimizer, 'load_and_export': ''},
            {"loss": losses.ArcfaceLoss(scale=64), "epoch": epoch, 'load_and_export': load_path},
        ]
    else:
        sch = [
            {"loss": losses.CosFaceLoss(scale=32), "epoch": 1, "optimizer": optimizer, 'load_and_export': ''},
            {"loss": losses.CosFaceLoss(scale=64), "epoch": epoch, 'load_and_export': load_path},
        ]
    tt.train(sch, 0)


if __name__ == "__main__":
    # for loss in ['arcface', 'cosface']:
    #     for cls in ['all', 'bird', 'cat', 'dog', 'small_animals']:
    for loss in ['arcface']:
        for cls in ['bird']:
            path = (f'/mnt/data/afarec/code/face_recognition/GhostFaceNets/'
                    f'work_dir_b256/{loss}_{cls}/ghostV2-1.3-1-({loss[0].upper()})_epoch44.h5')
            work_dir = f'/mnt/data/afarec/code/face_recognition/GhostFaceNets/work_dir_b256/{loss}_{cls}/'
            data_path = data_path_base + f'/split/{cls}/train.csv'
            export_from_epoch(path, 44)
