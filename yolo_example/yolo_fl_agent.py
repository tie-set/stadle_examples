import sys
import os
import time
from typing import List

import argparse
import subprocess
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from stadle import IntegratedClient
from stadle.lib.util.helpers import client_arg_parser

from yolov5.models.yolo import Model

def train(model, data, **kwargs):

    agg_save_dir = kwargs.get('agg_save_dir') if kwargs.get('agg_save_dir') else f'./output/eggplant_model/'
    data_path = kwargs.get('data_path') if kwargs.get('data_path') else f'./dataset/data_eggplant/eggplant_data.yaml'
    model_structure_dict_file = kwargs.get('model_structure_dict_file') if kwargs.get('model_structure_dict_file') else './dataset/weight/eggplant_1000.pt'

    epochs = int(kwargs.get("epochs")) if kwargs.get("epochs") else 2
    batch_size = int(kwargs.get("batch_size")) if kwargs.get("batch_size") else 2
    layers_frozen = int(kwargs.get("layers_frozen")) if kwargs.get("layers_frozen") else 0

    device = kwargs.get("device")

    def create_model_dict(model, model_structure_dict_file):
        model_structure_dict = torch.load(model_structure_dict_file)
        model_structure_dict['model'] = model
        return model_structure_dict


    merged_model_yolo_dict = create_model_dict(model, model_structure_dict_file)

    torch.save(merged_model_yolo_dict, (agg_save_dir + 'agg_model_weights.pt'))


    if (os.path.exists(agg_save_dir + "exp")):
        shutil.rmtree(agg_save_dir + "exp")

    yolo_cmd = ['python', 'yolov5/train.py', '--weights', (agg_save_dir + 'agg_model_weights.pt'), '--data', data_path,
                '--epochs', str(epochs), '--batch-size', str(batch_size), '--project', agg_save_dir,
                '--freeze', str(layers_frozen)]

    if (device):
        yolo_cmd.extend(['--device', device])

    yolo_process = subprocess.Popen(yolo_cmd)
    yolo_process.wait()

    trained_model_dict = torch.load(agg_save_dir + "/exp/weights/last.pt")

    shutil.rmtree(agg_save_dir + "exp")

    return trained_model_dict['model'], 0.0


def test(test_model, data, **kwargs):
    agg_save_dir = kwargs.get('agg_save_dir') if kwargs.get('agg_save_dir') else f'./output/eggplant_model/'
    data_path = kwargs.get('data_path') if kwargs.get('data_path') else f'./dataset/data_eggplant/eggplant_data.yaml'
    model_structure_dict_file = kwargs.get('model_structure_dict_file') if kwargs.get('model_structure_dict_file') else './dataset/weight/eggplant_1000.pt'

    device = kwargs.get("device")

    def create_model_dict(model, model_structure_dict_file):
        model_structure_dict = torch.load(model_structure_dict_file)
        model_structure_dict['model'] = model
        return model_structure_dict

    merged_model_yolo_dict = create_model_dict(test_model, model_structure_dict_file)

    torch.save(merged_model_yolo_dict, (agg_save_dir + 'test_model_weights.pt'))


    if (os.path.exists(agg_save_dir + "exp")):
        shutil.rmtree(agg_save_dir + "exp")

    yolo_cmd = ['python', 'yolov5/val.py', '--weights', (agg_save_dir + 'test_model_weights.pt'), '--data', data_path,
                '--project', agg_save_dir]
    if (device):
        yolo_cmd.extend(['--device', device])

    output = subprocess.run(yolo_cmd, capture_output=True)

    # Find line with mAP for all classes
    mAP_line = next(line for line in output.stdout.split(b'\n') if b'all' in line)
    mAP = float(mAP_line.split(b' ')[-1])

    shutil.rmtree(agg_save_dir + "exp")

    return mAP, 0

def validate(model, data, **kwargs):
    print("Validate Model")
    acc, ave_loss = test(test_model=model, data=data, **kwargs)
    print(f'Validation mAP@[0.5,0.95] of the model: {acc}')
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

    # Run forever
    if not kwargs.get("round_to_exit"):
        return True

    if current_fl_round >= int(kwargs.get("round_to_exit")):
        keep_running = False
        client.stop_model_exchange_routine()
    return keep_running

def get_model():
    model_structure_dict_file = './dataset/weight/eggplant_1000.pt'

    model = torch.load(model_structure_dict_file)['model']

    return model

if __name__ == '__main__':
    # YOLOv5-specific parameters for local training
    epochs = 1
    batch_size = 2
    layers_frozen = 0

    # global param

    train_round = 0

    parser = client_arg_parser()
    parser.add_argument('--dataset', type=str, help='Choose local dataset to train on (eggplant/negi/tomato)')
    parser.add_argument('--device', type=str, help='Specify device(s) to use when running YOLOv5 (ex: --device cpu, --device 0, --device 0,1,2,3)')
    parser.add_argument('--termination_round', type=int, help='Specify round to stop FL')
    parser.add_argument('--start_round', type=int, help='Specify round to join FL process', default=0)

    args = parser.parse_args()

    data_yaml = f'./dataset/data_{args.dataset}/{args.dataset}_data.yaml'
    agg_save_dir = f'./output/{args.dataset}_model/'

    if not os.path.exists(agg_save_dir):
        os.makedirs(agg_save_dir)




    integrated_client = IntegratedClient(config_file='config/config_agent.json', cl_args=args)
    integrated_client.maximum_rounds = 100000

    integrated_client.set_termination_function(judge_termination, round_to_exit=args.termination_round, client=integrated_client)

    integrated_client.set_training_function(train, None, agg_save_dir=agg_save_dir, data_path=data_yaml,
                                       model_structure_dict_file=model_structure_dict_file,
                                       epochs=epochs, batch_size=batch_size, layers_frozen=layers_frozen, device=args.device)

    integrated_client.set_validation_function(validate, None, agg_save_dir=agg_save_dir,
                                               data_path=data_yaml, model_structure_dict_file=model_structure_dict_file, device=args.device)
    integrated_client.set_testing_function(test, None)

    integrated_client.set_bm_obj(model)


    integrated_client.set_exch_active(False)

    time.sleep(5)

    while (integrated_client.round < args.start_round):
        time.sleep(2)

    integrated_client.set_exch_active(True)


    integrated_client.start()
    training_done = integrated_client.training_finalized
    if training_done:
        print("Training Completed!")
