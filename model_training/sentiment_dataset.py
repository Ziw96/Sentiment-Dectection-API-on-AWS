"""
Module holing dataset methods

Author pharnoux

"""

import os
import json
import math
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
import sagemaker

# s = sagemaker.Session()
# all_files= s.list_s3_files("asg4", "dev_features/")
# print(all_files)

def train_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "train")

def validation_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "validation")

def eval_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "eval")

def serving_input_fn(_, config):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _load_json_file(json_path, config):

    features = []
    labels = []

    if config["cloud"] == 0:
        with open(json_path, "r") as file:

            for line in file:

                entry = json.loads(line)

                if len(entry["features"]) != config["padding_size"]:
                    raise ValueError("The size of the features of the entry with twitterid {} was not expected".format(entry["twitterid"]))

                labels.append(entry["sentiment"] / 4)
                features.append(entry["features"])
    else:
        contents = json_path.split("\n")
        for line in contents:
            entry = json.loads(line)

            if len(entry["features"]) != config["padding_size"]:
                raise ValueError(
                    "The size of the features of the entry with twitterid {} was not expected".format(entry["twitterid"]))

            labels.append(entry["sentiment"] / 4)
            features.append(entry["features"])

    return features, labels

def _input_fn(directory, config, mode):

    print("Fetching {} data...".format(mode))
    all_features = []
    all_labels = []

    if config["cloud"] == 0:
        all_files = os.listdir(directory)
        for file in all_files:
            features, labels = _load_json_file(os.path.join(directory, file), config)
            all_features += features
            all_labels += labels
    else:
        sess = sagemaker.Session()
        all_files = sess.list_s3_files(config["bucket"], directory)
        for file in all_files[1:]:
            features, labels = _load_json_file(sess.read_s3_file(config["bucket"], file), config)
            all_features += features
            all_labels += labels

    num_data_points = len(all_features)
    num_batches = math.ceil(len(all_features) / config["batch_size"])

    dataset = Dataset.from_tensor_slices((all_features, all_labels))

    if mode == "train":

        dataset = Dataset.from_tensor_slices((all_features, all_labels))
        dataset = dataset.batch(config["batch_size"]).shuffle(10000, seed=12345).repeat(
            config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    if mode in ("validation", "eval"):

        dataset = dataset.batch(config["batch_size"]).repeat(config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    iterator = dataset.make_one_shot_iterator()
    dataset_features, dataset_labels = iterator.get_next()

    return [{config["input_tensor_name"]: dataset_features}, dataset_labels,
            {"num_data_point": num_data_points, "num_batches": num_batches}]
