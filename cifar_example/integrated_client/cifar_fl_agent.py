import os
import sys
from typing import List
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from stadle import IntegratedClient

from vgg import VGG as Model

def data_processing(data_save_path: str = "./data", max_workers=2, batch_size=64, args=None):

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_save_path, train=True, download=True, transform=transform_train)

    ###
    # Modification for imbalanced train datasets

    class_counts = int(args.def_count * 5000) * np.ones(len(classes))

    for c in classes:
        if getattr(args, c):
            class_counts[trainset.class_to_idx[c]] = int(args.sel_count * 5000)

    class_counts_ref = np.copy(class_counts)

    imbalanced_idx = []

    for i,img in enumerate(trainset):
        c = img[1]
        if (class_counts[c] > 0):
            imbalanced_idx.append(i)
            class_counts[c] -= 1

    trainset = torch.utils.data.Subset(trainset, imbalanced_idx)

    ###

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=max_workers)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=max_workers)

    return trainloader, testloader

def train(model, data, **kwargs):
    lr = float(kwargs.get("lr")) if kwargs.get("lr") else 0.001
    momentum = float(kwargs.get("momentum")) if kwargs.get("momentum") else 0.9
    epochs = int(kwargs.get("epochs")) if kwargs.get("epochs") else 2
    device = kwargs.get("device") if kwargs.get("device") else 'cpu'

    agent_name = kwargs.get("agent_name") if kwargs.get("agent_name") else 'default_agent'

    print('Saving..')
    if not os.path.isdir(f'checkpoint/{agent_name}'):
        os.makedirs(f'checkpoint/{agent_name}', exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoint/{agent_name}/ckpt_sg.pth")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    ave_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        print('\nEpoch: %d' % (epoch + 1))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            ave_loss.append(train_loss)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write('\r'+f"\rEpoch Accuracy: {(100*correct/total):.2f}%")
        print('\n')
    print('Finished Training')
    ave_loss = sum(ave_loss) / len(ave_loss)

    model = model.to('cpu')

    print('Saving..')
    if not os.path.isdir(f'checkpoint/{agent_name}'):
        os.makedirs(f'checkpoint/{agent_name}', exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoint/{agent_name}/ckpt_local.pth")

    return model, ave_loss


def test(test_model, data, **kwargs):
    device = kwargs.get("device") if kwargs.get("device") else 'cpu'

    test_model = test_model.to(device)

    correct = 0
    total = 0
    overall_accuracy = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (inputs, targets) in data:
            inputs, targets = inputs.to(device), targets.to(device)
            # calculate outputs by running images through the network
            outputs = test_model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    overall_accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (overall_accuracy))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for (inputs, targets) in data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = test_model(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for target, prediction in zip(targets, predictions):
                if prediction == target:
                    correct_pred[classes[target]] += 1
                total_pred[classes[target]] += 1

    # print accuracy for each class
    # Capture average accuracy across all classes
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
    return overall_accuracy, 0


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
    # MLFLow

    parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

    parser.add_argument('--def_count', default=0.1, type=float)
    parser.add_argument('--sel_count', default=1.0, type=float)

    parser.add_argument('--lt_epochs', default=3)

    parser.add_argument('--agent_name', default='default_agent')

    parser.add_argument('--cuda', action='store_true', default=False)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    for c in classes:
        parser.add_argument(f'--{c}', action='store_true', default=False)

    args = parser.parse_args()

    sel_classes = [c for c in classes if getattr(args, c)]

    config_file = 'config/config_agent.json'

    device = 'cuda:0' if args.cuda else 'cpu'

    trainloader, testloader = data_processing(data_save_path="./data", args=args)

    model = Model('VGG16')

    integrated_client = IntegratedClient(config_file=config_file, simulation_flag=True, agent_name=args.agent_name)
    integrated_client.maximum_rounds = 100000

    integrated_client.set_termination_function(judge_termination, round_to_exit=20, client=integrated_client)

    integrated_client.set_training_function(train, trainloader, lr=args.lr, epochs=args.lt_epochs, device=device, agent_name=args.agent_name)
    integrated_client.set_validation_function(validate, testloader, device=device)
    integrated_client.set_testing_function(test, testloader)

    integrated_client.set_bm_obj(model)

    integrated_client.start()
    training_done = integrated_client.training_finalized
    if training_done:
        print("Training Completed!")
