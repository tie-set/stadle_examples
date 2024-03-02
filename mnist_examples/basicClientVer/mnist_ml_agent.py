import argparse
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MyNet

from stadle import BasicClient


def load_MNIST(batch=128, intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    return {'train': train_loader, 'test': test_loader}


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='STADLE MNIST Training')
    parser.add_argument('--agent_name', default='default_agent')
    args = parser.parse_args()
    
    # traing count number
    epoch = 20

    # trainging result
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    # create network
    net: torch.nn.Module = MyNet()

    # get ladaloer mnist
    loaders = load_MNIST()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    # create stadle_client
    client_config_path = r'client_config.json'
    stadle_client = BasicClient(config_file=client_config_path,agent_name=args.agent_name)

    # create instance
    stadle_client.set_bm_obj(net)

    for e in range(epoch):

        # stadle model
        if (e % 2 == 0):

            # Don't send model at beginning of training
            if (e != 0):
                perf_dict = {
                            'performance':history['test_acc'][-1],
                            'accuracy' : history['test_acc'][-1],
                             'loss_training' : history['train_loss'][-1],
                             'loss_test' : history['test_loss'][-1]}
                stadle_client.send_trained_model(net, perf_dict)

            state_dict = stadle_client.wait_for_sg_model().state_dict()
            net.load_state_dict(state_dict)


        """ Training Part"""
        loss = None

        # start training
        net.train(True)
        for i, (data, target) in enumerate(loaders['train']):
            data = data.view(-1, 28 * 28)
            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch ({} / 60000 train. data). Loss: {}'.format(e + 1, (i + 1) * 128,
                                                                                         loss.item()))

        history['train_loss'].append(loss.item())

        """ Test Part """
        # stop training
        net.eval()  # or net.train(False)
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 28 * 28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 10000

        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss, correct / 10000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)

