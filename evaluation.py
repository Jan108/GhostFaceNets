import argparse
import os
import time
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import GhostFaceNets
from data import load_petface_verification, load_petface_identification


def verification(params):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.mixed_precision.set_global_policy("mixed_float16")

    # Load model GhostFaceNetV2 Strides 1
    basic_model = GhostFaceNets.buildin_models("ghostnetv2", dropout=0, emb_shape=512,
                                               output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
    basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4,
                                                           apply_to_batch_normal=False)
    basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)
    basic_model.load_weights(params.weights)

    def net(img):
        feat = basic_model(img)
        feat_f = basic_model(tf.image.flip_left_right(img))
        return feat + feat_f

    # Load data
    ds, img1_list, img2_list = load_petface_verification(params.img_path, params.img_verification, batch_size=128)

    # Predict similarity
    sim_list = []
    label_list = []
    for img1, img2, label in tqdm(ds, desc='Test image pairs'):
        vec1 = tf.nn.l2_normalize(net(img1), axis=1)
        vec2 = tf.nn.l2_normalize(net(img2), axis=1)

        sim = tf.reduce_sum(vec1 * vec2, axis=1)
        sim_list.extend(sim.numpy().tolist())
        label_list.extend(label.numpy().tolist())

    os.makedirs(os.path.dirname(params.output), exist_ok=True)
    pd.DataFrame(
        {'file1': img1_list, 'file2': img2_list,
         'sim': sim_list, 'label': label_list}).to_csv(
        os.path.join(params.output, 'verification.csv'), index=False)

    # Test latency
    ds, _, _ = load_petface_verification(params.img_path, params.img_verification, batch_size=1)

    inf_times = []
    for img1, img2, label in tqdm(islice(ds, params.latency_test),
                                  desc='Test image pairs latency', total=params.latency_test):
        start_time = datetime.now()
        tf.nn.l2_normalize(net(img1), axis=1)
        tf.nn.l2_normalize(net(img2), axis=1)
        end_time = datetime.now()
        inf_times.append(end_time - start_time)
    avg_time = sum(inf_times, start=timedelta()) / len(inf_times)
    print(f'Inference for {params.weights} took {avg_time}')
    with open(os.path.join(params.output, 'timing.txt'), 'w') as file:
        file.write(str(avg_time))


def identification(params):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.mixed_precision.set_global_policy("mixed_float16")

    pool_ds = load_petface_identification(params.img_path, params.img_identification, pool_only=True, batch_size=128)
    test_ds = load_petface_identification(params.img_path, params.img_identification, pool_only=False, batch_size=64)

    # Load model GhostFaceNetV2 Strides 1
    basic_model = GhostFaceNets.buildin_models("ghostnetv2", dropout=0, emb_shape=512,
                                               output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
    basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4,
                                                           apply_to_batch_normal=False)
    basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)
    basic_model.load_weights(params.weights)

    def net(img):
        feat = basic_model(img)
        feat_f = basic_model(tf.image.flip_left_right(img))
        return feat + feat_f

    features, labels = [], []
    for images, batch_labels in tqdm(pool_ds, desc="Extracting features"):
        batch_features = tf.nn.l2_normalize(net(images), axis=1)
        features.append(batch_features)
        labels.extend(batch_labels.numpy().tolist())

    pool_features, pool_labels = tf.concat(features, axis=0), labels

    # Open output file
    os.makedirs(os.path.dirname(params.output), exist_ok=True)
    with open(os.path.join(params.output, params.identification_file), 'w') as f:
        f.write(f"test_label,{','.join([f'predicted_label_{i}' for i in range(5)])},{','.join([f'similarity_{i}' for i in range(5)])}\n")

        # Process test images in batches
        for test_images, test_labels in tqdm(test_ds, desc="Processing test images"):
            test_features = tf.nn.l2_normalize(net(test_images), axis=1)

            # Compute cosine similarity (dot product of normalized vectors)
            sim = tf.matmul(test_features, pool_features, transpose_b=True)  # (batch_size, num_pool)
            top5_sim, top5_indices = tf.math.top_k(sim, k=5)
            # Write results
            for test_label, pred_label, pred_sim in zip(test_labels.numpy().tolist(), top5_indices.numpy().tolist(), top5_sim.numpy().tolist()):
                f.write(f"{test_label},{','.join([str(i) for i in pred_label])},{','.join([str(i) for i in pred_sim])}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow GhostFaceNet Evaluation')
    parser.add_argument("--output", type=str, default='./work_dir', help="Output directory")
    parser.add_argument("--weights", type=str, default='./work_dir', help="Model weights")
    parser.add_argument("--img_path", type=str, default='./data/images', help="Image file rott directory")
    parser.add_argument("--img_verification", type=str, default='./data/split/verification.csv',
                        help="File containing a list of images files with corresponding label for verification")
    parser.add_argument("--img_identification", type=str, default='./data/split/identification.csv',
                        help="File containing a list of images files with corresponding label for identification")
    parser.add_argument("--latency_test", type=int, default=1000, help="Amount of images for the latency test")
    parser.add_argument("--ident-general", action='store_true', help="Generalized model evaluation")
    args = parser.parse_args()
    # verification(args)
    args.identification_file = 'identification.csv'
    if args.ident_general:
        base_path_ident = args.img_identification
        print('Loading and performing each class separately for generalized model')
        for cls in ['bird', 'cat', 'dog', 'small_animals']:
            args.img_identification = f'{base_path_ident}/{cls}/identification_img.csv'
            args.identification_file = f'identification_{cls}.csv'
            identification(args)
    else:
        identification(args)
