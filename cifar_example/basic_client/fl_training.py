import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from vgg import VGG

from stadle import BasicClient


def data_processing(data_save_path='./cifar_data', batch_size=64, cl_args=None):
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
        if c in args.classes:
            class_counts[trainset.class_to_idx[c]] = int(args.sel_count * 5000)

    imbalanced_idx = []

    for i,img in enumerate(trainset):
        c = img[1]
        if (class_counts[c] > 0):
            imbalanced_idx.append(i)
            class_counts[c] -= 1

    trainset = torch.utils.data.Subset(trainset, imbalanced_idx)

    ###

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=data_save_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def local_training(model, optimizer, device, num_epochs=1):
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    train_loss = 0
    correct = 0
    total = 0

    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write('\r'+f"\rEpoch Accuracy: {(100*correct/total):.2f}%")
        print('\n')

    train_acc = correct/total

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    print(f"Accuracy on validation set: {(100*test_acc):.2f}%")

    return train_loss / (len(trainloader) * num_epochs), train_acc, test_loss / len(testloader), test_acc


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='STADLE CIFAR-10 Training')
    parser.add_argument('--agent_name', default='pytorch_agent')
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--def_count', default=0.1, type=float)
    parser.add_argument('--sel_count', default=1.0, type=float)
    parser.add_argument('--classes', default=[], nargs='*')

    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    client_config_file = 'client_config.json'
    stadle_client = BasicClient(config_file=client_config_file, agent_name=args.agent_name)

    trainloader, testloader = data_processing(cl_args=args)

    model = VGG('VGG16')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for rnd in range(args.num_rounds):
        # Update the local model with the aggregate model weights
        state_dict = stadle_client.wait_for_sg_model().state_dict()
        model.load_state_dict(state_dict)

        train_loss, train_acc, valid_loss, valid_acc = local_training(model, optimizer, args.device, num_epochs=2)
        
        perf_metrics = {
            'performance': train_acc,
            'accuracy': valid_acc,
            'loss_training': train_loss,
            'loss_test': valid_loss,
        }

        # Send the locally trained model with the associated performance metric values
        stadle_client.send_trained_model(model, perf_values=perf_metrics)
        
        print(f'Sending model for round {rnd+1} to aggregator')

    stadle_client.disconnect()