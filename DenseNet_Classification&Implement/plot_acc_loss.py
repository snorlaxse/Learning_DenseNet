'''
@FileName: plot_acc_loss.py
@Author: CaptainSE
@Time: 2019-03-01 
@Desc: 

'''

import matplotlib.pyplot as plt

def plot_acc_loss(loss_values, val_loss_values, acc, val_acc):

    # 绘制训练损失和验证损失
    epochs = range(1, len(loss_values) + 1)
    # 'bo' 蓝色圆点
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # 'b' 蓝色实线
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title("Training and Validation loss")
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # 绘制训练精度和验证精度
    # plt.clf() 清空图像


    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()