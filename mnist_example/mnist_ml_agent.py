import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
from models.mynet import MyNet

from stadle import BasicClient

import argparse


def load_MNIST(batch=128, intensity=1.0, classes=None, sel_prob=1.0, def_prob=0.1):
    trainset_size = 60000

    if (args.classes is not None):
        trainset = datasets.MNIST('./data',
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x * intensity)
                            ])
                          )
        classes = [int(c) for c in args.classes.split(',')]

        mask = (trainset.targets == -1)

        for i in range(10):
            class_mask = (trainset.targets == i)
            mask_idx = class_mask.nonzero()
            class_size = len(mask_idx)

            size = sel_prob if (i in classes) else def_prob

            mask_idx = mask_idx[torch.randperm(class_size)][:int(class_size * size)]

            mask[mask_idx] = True
            

        trainset.data = trainset.data[mask]
        trainset.targets = trainset.targets[mask]

        trainset_size = len(trainset)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)

    else:
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

    return {'train': train_loader, 'test': test_loader}, trainset_size


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')
    parser.add_argument('--agent_name', default='default_agent')
    parser.add_argument('--classes')
    parser.add_argument('--def_prob', type=float, default=0.1)
    parser.add_argument('--sel_prob', type=float, default=1.0)
    args = parser.parse_args()

    # 学習回数
    epoch = 20

    # 学習結果の保存用
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    # ネットワークを構築
    net: torch.nn.Module = MyNet()

    # MNISTのデータローダーを取得
    loaders, trainset_size = load_MNIST(classes=args.classes, def_prob=args.def_prob, sel_prob=args.sel_prob)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    client_config_path = r'config/config_agent.json'
    stadle_client = BasicClient(config_file=client_config_path, agent_name=args.agent_name)
    # インスタンス化して入れる
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
        # 学習開始 (再開)
        net.train(True)  # 引数は省略可能

        for i, (data, target) in enumerate(loaders['train']):
            # 全結合のみのネットワークでは入力を1次元に
            # print(data.shape)  # torch.Size([128, 1, 28, 28])
            data = data.view(-1, 28 * 28)
            # print(data.shape)  # torch.Size([128, 784])

            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch ({} / {} train. data). Loss: {}'.format(e + 1, (i + 1) * 128,
                                                                                         trainset_size, loss.item()))

        history['train_loss'].append(loss.item())

        """ Test Part """
        # 学習のストップ
        net.eval()  # または net.train(False) でも良い
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

        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)

    # 結果の出力と描画
    print(history)
    plt.figure()
    plt.plot(range(1, epoch + 1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch + 1), history['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(range(1, epoch + 1), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('test_acc.png')
