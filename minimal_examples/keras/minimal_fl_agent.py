import os
import sys
import argparse
import time
import random
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from stadle import IntegratedClient
from stadle.lib.util.helpers import client_arg_parser


# from minimal_model import MinimalModel

def get_minimal_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=(3,)),
        keras.layers.Dense(4)
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model

def train(model, data, **kwargs):
    # Sleep for 15 seconds to mimic local training
    # This would be replaced by model training code
    time.sleep(15)
    print("Completed training")

    # Set trained model to original model for minimal example
    trained_model = model
    # Example metric (average loss) that the training function can return
    ave_loss = 0

    return trained_model, ave_loss


def test(test_model, data, **kwargs):
    # Performance computation is not included in minimal example - the
    # implementation of this method is dependent on the data and model
    # that is being trained.

    # Example performance metrics (accuracy and average loss) that can
    # be computed
    acc = random.random()
    ave_loss = 10 * random.random()

    return acc, ave_loss


def validate(model, data, **kwargs):
    print("Validate Model")
    acc, ave_loss = test(test_model=model, data=data, **kwargs)
    print(f'Validation Accuracy of the model: {acc} %')
    return acc, ave_loss


def judge_termination(**kwargs) -> bool:
    """
    Decide if it finishes training process and exits from FL platform
    :param training_count: int - the number of training done
    :param sg_arrival_count: int - the number of times it received SG models
    :return: bool - True if it continues the training loop; False if it stops
    """

    # Depending on the criteria for termination,
    # change the return bool value

    keep_running = True
    client = kwargs.get('client')
    current_fl_round = client.federated_training_round
    print(f"Current Federated Learning Round: >>>>>> : {current_fl_round}")
    if current_fl_round >= int(kwargs.get("round_to_exit")):
        keep_running = False
        client.stop_model_exchange_routine()
    return keep_running


if __name__ == '__main__':
    client_config_file = 'config/config_agent.json'

    model = get_minimal_model()

    integrated_client = IntegratedClient(config_file=client_config_file, cl_args=client_arg_parser().parse_args())
    integrated_client.maximum_rounds = 100000

    integrated_client.set_termination_function(judge_termination, round_to_exit=20, client=integrated_client)

    integrated_client.set_training_function(train, None)
    integrated_client.set_validation_function(validate, None)
    integrated_client.set_testing_function(test, None)

    integrated_client.set_bm_obj(model)

    integrated_client.start()
    training_done = integrated_client.training_finalized
    if training_done:
        print("Training Completed!")
