# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import torch.autograd
from tqdm import tqdm
__all__ = ['training', 'testing']


def training(opt, writer, train_loader, test_loader, net, pre_epoch, device):
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)

    for epoch in range(opt.NUM_EPOCHS):
        train_loss = 0
        train_acc  = 0

        # Start training
        net.train()
        print('==> Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader), leave=False, unit='b'):
            inputs, labels, *_ = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = opt.CRITERION(outputs, labels)
            predicts = outputs.sort(descending=True)[1][:, :opt.TOP_NUM]
            for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
                if label in predict:
                    train_acc += 1

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / opt.NUM_TRAIN
        train_acc = train_acc / opt.NUM_TRAIN

        # Start testing
        test_loss, test_acc = testing(opt, test_loader, net, device)

        # Add summary to tensorboard
        writer.add_scalar("Train/loss", train_loss, epoch + pre_epoch)
        writer.add_scalar("Train/acc", train_acc, epoch + pre_epoch)
        writer.add_scalar("Test/loss", test_loss, epoch + pre_epoch)
        writer.add_scalar("Test/acc", test_acc, epoch + pre_epoch)

        # Output results
        print('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc:%.4f'
              % (pre_epoch + epoch + 1, pre_epoch + opt.NUM_EPOCHS + 1, train_loss, train_acc, test_loss, test_acc))

        # Save the model
        if epoch % opt.SAVE_EVERY == 0:
            net.mt_save(pre_epoch + epoch + 1, test_loss / opt.NUM_TEST)

    print('==> Training Finished.')
    return net


def testing(opt, test_loader, net, device):
    net.eval()
    test_loss = 0
    test_acc = 0
    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = opt.CRITERION(outputs, labels)
        test_loss += loss.item()

        predicts = outputs.sort(descending=True)[1][:, :opt.TOP_NUM]
        for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
            if label in predict:
                test_acc += 1
    return test_loss / opt.NUM_TEST, test_acc / opt.NUM_TEST
