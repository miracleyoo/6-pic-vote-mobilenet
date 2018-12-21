#!/usr/bin/python
# -*- coding: UTF-8 -*-

import threading
import time

exitFlag = 0
best_loss = 0


def save_models(opt, net, epoch, train_loss, best_loss, test_loss):
    # Save a temp model
    if opt.SAVE_TEMP_MODEL:
        net.save(epoch, train_loss / opt.NUM_TRAIN, "temp_model.dat")

    # Save the best model
    if test_loss / opt.NUM_TEST < best_loss:
        best_loss = test_loss / opt.NUM_TEST
        net.save(epoch, train_loss / opt.NUM_TRAIN, "best_model.dat")

    return best_loss


class MyThread(threading.Thread):
    def __init__(self, opt, net, epoch, train_loss, best_loss, test_loss):
        threading.Thread.__init__(self)
        self.opt = opt
        self.net = net
        self.epoch = epoch
        self.train_loss = train_loss
        self.best_loss = best_loss
        self.test_loss = test_loss

    def run(self):
        global best_loss
        lock.acquire()
        try:
            best_loss = save_models(self.opt, self.net, self.epoch, self.train_loss, self.best_loss, self.test_loss)
        finally:
            # 改完了一定要释放锁:
            lock.release()


def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        lock.acquire()
        try:
            print("%s: %s" % (threadName, time.ctime(time.time())))
            # while True: pass
        finally:
            # 改完了一定要释放锁:
            lock.release()
        counter -= 1


lock = threading.Lock()
threads = []

# 创建新线程
thread1 = MyThread(1, "Thread-1", 0.5)
thread2 = MyThread(2, "Thread-2", 1)

# 开启新线程
thread1.start()
thread2.start()
print_time('Main', 1.5, 2)
# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)

# 等待所有线程完成
for t in threads:
    t.join()
print("Exiting Main Thread")