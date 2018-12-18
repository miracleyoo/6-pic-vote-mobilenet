# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import codecs
import datetime
import os
import pickle
import shutil
import socket
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

lock = threading.Lock()


def log(*args, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]),
              end=end)


def to_multi(net):
    """
    If you have multiple GPUs and you want to use them at the same time, you should
    call this method before training to send your model and data to multiple GPUs.
    :return: None
    """
    if torch.cuda.is_available():
        log("Using", torch.cuda.device_count(), "GPUs.")
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
            attrs_p = [meth for meth in dir(net) if not meth.startswith('_')]
            attrs = [meth for meth in dir(net.module) if not meth.startswith('_') and meth not in attrs_p]
            for attr in attrs:
                setattr(net, attr, getattr(net.module, attr))
            log("Using data parallelism.")
    else:
        log("Using CPU now.")
    net.to(net.device)
    return net


class MyThread(threading.Thread):
    """
        Multi-thread support class. Used for multi-thread model
        file saving.
    """

    def __init__(self, opt, net, epoch, bs_old, loss):
        threading.Thread.__init__(self)
        self.opt = opt
        self.net = net
        self.epoch = epoch
        self.bs_old = bs_old
        self.loss = loss

    def run(self):
        lock.acquire()
        try:
            if self.opt.SAVE_TEMP_MODEL:
                self.net.save(self.epoch, self.loss, "temp_model.dat")
            if self.opt.SAVE_BEST_MODEL and self.loss < self.bs_old:
                self.net.best_loss = self.loss
                net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID + '/'
                temp_model_name = net_save_prefix + "temp_model.dat"
                best_model_name = net_save_prefix + "best_model.dat"
                shutil.copy(temp_model_name, best_model_name)
        finally:
            lock.release()


class BasicModule(nn.Module):
    """
        Basic pytorch module class. A wrapped basic model class for pytorch models.
        You can Inherit it to make your model easier to use. It contains methods
        such as load, save, multi-thread save, parallel distribution, train, validate,
        predict and so on.
    """

    def __init__(self, opt=None, device=None):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__
        self.opt = opt
        self.best_loss = 1e8
        self.epoch_fin = 0
        self.threads = []
        self.server_name = socket.getfqdn(socket.gethostname())
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': [], 'epoch': 0}
        self.writer = SummaryWriter(opt.SUMMARY_PATH)

    def load(self, model_type: str = "temp_model.dat", map_location=None) -> None:
        """
            Load the existing model.
            :param model_type: temp model or best model.
            :param map_location: your working environment.
                For loading the model file dumped from gpu or cpu.
            :return: None.
        """
        log('Now using ' + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID)
        log('Loading model ...')
        if not map_location:
            map_location = self.device.type
        net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID + '/'
        temp_model_name = net_save_prefix + model_type
        if not os.path.exists(net_save_prefix):
            os.mkdir(net_save_prefix)
        if os.path.exists(temp_model_name):
            checkpoint = torch.load(temp_model_name, map_location=map_location)
            self.epoch_fin = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.history = checkpoint['history']
            self.load_state_dict(checkpoint['state_dict'])
            log("Load existing model: %s" % temp_model_name)
        else:
            log("The model you want to load (%s) doesn't exist!" % temp_model_name)

    def save(self, epoch, loss, name=None):
        """
        Save the current model.
        :param epoch:The current epoch (sum up). This will be together saved to file,
            aimed to keep tensorboard curve a continuous line when you train the net
            several times.
        :param loss:Current loss.
        :param name:The name of your saving file.
        :return:None
        """
        if loss < self.best_loss:
            self.best_loss = loss
        if self.opt is None:
            prefix = "./source/trained_net/" + self.model_name + "/"
        else:
            prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + \
                     self.opt.PROCESS_ID + '/'
            if not os.path.exists(prefix): os.mkdir(prefix)

        if name is None:
            name = "temp_model.dat"

        path = prefix + name
        try:
            state_dict = self.module.state_dict()
        except:
            state_dict = self.state_dict()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'best_loss': self.best_loss,
            'history': self.history
        }, path)

    def mt_save(self, epoch, loss):
        """
        Save the model with a new thread. You can use this method in stead of self.save to
        save your model while not interrupting the training process, since saving big file
        is a time-consuming task.
        Also, this method will automatically record your best model and make a copy of it.
        :param epoch: Current loss.
        :param loss:
        :return: None
        """
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            log("Your best model is renewed")
        if len(self.threads) > 0:
            self.threads[-1].join()
        self.threads.append(MyThread(self.opt, self, epoch, self.best_loss, loss))
        self.threads[-1].start()
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            log("Your best model is renewed")
            self.best_loss = loss

    def _get_optimizer(self):
        """
        Get your optimizer by parsing your opts.
        :return:Optimizer.
        """
        if self.opt.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.LEARNING_RATE)
        else:
            raise KeyError("==> The optimizer defined in your config file is not supported!")
        return optimizer

    def to_multi(self):
        """
        If you have multiple GPUs and you want to use them at the same time, you should
        call this method before training to send your model and data to multiple GPUs.
        :return: None
        """
        if torch.cuda.is_available():
            log("Using", torch.cuda.device_count(), "GPUs.")
            if torch.cuda.device_count() > 1:
                pmodel = torch.nn.DataParallel(self)
                attrs_p = [meth for meth in dir(pmodel) if not meth.startswith('_')]
                attrs = [meth for meth in dir(self) if not meth.startswith('_') and meth not in attrs_p]
                for attr in attrs:
                    setattr(pmodel, attr, getattr(self, attr))
                log("Using data parallelism.")
        else:
            log("Using CPU now.")
        pmodel.to(self.device)
        return pmodel

    def validate(self, val_loader):
        """
        Validate your model.
        :param val_loader: A DataLoader class instance, which includes your validation data.
        :return: val loss and val accuracy.
        """
        self.eval()
        val_loss = 0
        val_acc = 0
        for i, data in tqdm(enumerate(val_loader), desc="Validating", total=len(val_loader), leave=False, unit='b'):
            inputs, labels, *_ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Compute the outputs and judge correct
            outputs = self(inputs)
            loss = self.opt.CRITERION(outputs, labels)
            val_loss += loss.item()

            predicts = outputs.sort(descending=True)[1][:, :self.opt.TOP_NUM]
            for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
                if label in predict:
                    val_acc += 1
        return val_loss / self.opt.NUM_VAL, val_acc / self.opt.NUM_VAL

    def predict(self, val_loader):
        """
        Make prediction based on your trained model. Please make sure you have trained
        your model or load the previous model from file.
        :param test_loader: A DataLoader class instance, which includes your test data.
        :return: Prediction made.
        """
        recorder = []
        log("Start predicting...")
        self.eval()
        for i, data in tqdm(enumerate(val_loader), desc="Validating", total=len(val_loader), leave=False, unit='b'):
            inputs, *_ = data
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            predicts = outputs.sort(descending=True)[1][:, :self.opt.TOP_NUM]
            recorder.extend(np.array(outputs.sort(descending=True)[1]))
            pickle.dump(np.concatenate(recorder, 0), open("./source/test_res.pkl", "wb+"))
        return predicts

    def vote_val(self, val_loader):
        log("Start vote predicting...")
        self.eval()
        val_loss = 0
        val_acc = 0

        def mode(x, x_vals):
            unique, counts = np.unique(x, return_counts=True)
            max_pos = np.where(counts == counts.max())[0]
            print(max_pos)
            if len(counts) >= 2 and len(max_pos) > 1:
                res = np.array([np.where(x_vals == unique[max_pos[i]])[0].max() for i in range(len(max_pos))])
                max_index = res.argmax()
                return unique[max_pos[max_index]]
            else:
                return unique[counts.argmax()]

        for i, data in enumerate(val_loader):
            inputs, labels, *_ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self(inputs)
            loss = self.opt.CRITERION(outputs, labels)
            val_loss += loss.item()
            label = labels.detach().tolist()[0]

            predicts = outputs.sort(descending=True)[1][:, 0].detach().cpu().numpy()
            pred_vals = outputs.sort(descending=True)[0][:, 0].detach().cpu().numpy()
            valid_voters = pred_vals.argsort()[::-1][:5]
            valid_votes = predicts[valid_voters]
            valid_vals = pred_vals[valid_voters]

            res = mode(valid_votes, valid_vals)
            if res == -1:
                res = predicts[pred_vals.argmax()]

            print(res == label, res, label, valid_voters, valid_votes, pred_vals[valid_voters], predicts)

            if label == res:
                val_acc += 1

        log("val_acc:{}".format(val_acc / self.opt.NUM_VAL))

    def fit(self, train_loader, val_loader):
        """
        Training process. You can use this function to train your model. All configurations
        are defined and can be modified in config.py.
        :param train_loader: A DataLoader class instance, which includes your train data.
        :param val_loader: A DataLoader class instance, which includes your test data.
        :return: None.
        """
        log("Start training...")
        epoch = 0
        optimizer = self._get_optimizer()
        for epoch in range(self.opt.NUM_EPOCHS):
            train_loss = 0
            train_acc = 0

            # Start training
            self.train()
            log('Preparing Data ...')
            for i, data in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader), leave=False,
                                unit='b'):
                inputs, labels, *_ = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                # print("Outside: input size", inputs.size(),
                #       "output_size", outputs.size())
                loss = self.opt.CRITERION(outputs, labels)
                predicts = outputs.sort(descending=True)[1][:, :self.opt.TOP_NUM]
                for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
                    if label in predict:
                        train_acc += 1

                # loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / self.opt.NUM_TRAIN
            train_acc = train_acc / self.opt.NUM_TRAIN

            # Start testing
            val_loss, val_acc = self.validate(val_loader)

            # Add summary to tensorboard
            self.writer.add_scalar("Train/loss", train_loss, epoch + self.epoch_fin)
            self.writer.add_scalar("Train/acc", train_acc, epoch + self.epoch_fin)
            self.writer.add_scalar("Eval/loss", val_loss, epoch + self.epoch_fin)
            self.writer.add_scalar("Eval/acc", val_acc, epoch + self.epoch_fin)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Output results
            log('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Eval Loss: %.4f, Eval Acc:%.4f'
                % (self.epoch_fin + epoch + 1, self.epoch_fin + self.opt.NUM_EPOCHS,
                   train_loss, train_acc, val_loss, val_acc))

            # Save the model
            if epoch % self.opt.SAVE_PER_EPOCH == 0:
                self.mt_save(self.epoch_fin + epoch + 1, val_loss / self.opt.NUM_VAL)

        self.epoch_fin = self.epoch_fin + epoch + 1
        # self.plot_history()
        self.write_summary()
        log('Training Finished.')

    def plot_history(self, figsize=(20, 9)):
        import matplotlib.pyplot as plt
        import seaborn as sns
        f, axes = plt.subplots(1, 2, figsize=figsize)
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['train_acc'], label='Train Accuracy', ax=axes[0])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['val_acc'], label='Val Accuracy', ax=axes[0])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['train_loss'], label='Train Loss', ax=axes[1])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['val_loss'], label='Val Loss', ax=axes[1])
        plt.tight_layout()
        if hasattr(self.opt, 'RUNNING_ON_JUPYTER') and self.opt.RUNNING_ON_JUPYTER:
            plt.show()
        else:
            f.savefig(os.path.join(self.opt.SUMMARY_PATH + "history_output.jpg"))

    def write_summary(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        sum_path = os.path.join(self.opt.SUMMARY_PATH, 'Model_Record_Form.md')
        with codecs.open('./config.py', 'r', encoding='utf-8') as f:
            raw_data = f.readlines()
            configs = "|Config Name|Value|\n|---|---|\n"
            for line in raw_data:
                if line.strip().startswith('self.'):
                    pairs = line.strip().lstrip('self.').split('=')
                    configs += '|' + pairs[0] + '|' + pairs[1] + '|\n'
        with codecs.open('./models/Template.txt', 'r', encoding='utf-8') as f:
            template = ''.join(f.readlines())

        try:
            content = template % (
                self.model_name,
                current_time,
                self.server_name,
                self.history['epoch'],
                max(self.history['val_acc']),
                sum(self.history['val_acc']) / len(self.history['val_acc']),
                sum(self.history['val_loss']) / len(self.history['val_loss']),
                sum(self.history['train_acc']) / len(self.history['train_acc']),
                sum(self.history['train_loss']) / len(self.history['train_loss']),
                os.path.basename(self.opt.TRAIN_PATH),
                os.path.basename(self.opt.EVAL_PATH),
                self.opt.NUM_CLASSES,
                self.opt.CRITERION.__class__.__name__,
                self.opt.OPTIMIZER,
                self.opt.LEARNING_RATE,
                configs,
                str(self)
            )
            with codecs.open(sum_path, 'w+', encoding='utf-8') as f:
                f.writelines(content)
        except:
            raise KeyError("Template doesn't exist or it conflicts with your format.")
