import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import make_model
from config import Config
from utils import load_graphdata

config = Config()
lr = config.lr
epochs = config.epochs
batch_size = config.batch_size
num_layers = config.num_layers
d_model = config.d_model
n_head = config.n_head
dropout = config.dropout
kernel_size = config.kernel_size
smooth_layer = config.smooth_layer_num
encoder_input_size = config.encoder_input_size
decoder_output_size = config.decoder_output_size
num_for_predict = config.num_for_predict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adj_mx = np.load(config.adj_filename)
params_path = config.params_path
graph_signal_matrix_filename = config.matrix_file

train_loader, _, val_loader, _, max, min = load_graphdata(graph_signal_matrix_filename, DEVICE,
                                                             batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, n_head, num_for_predict,
                 dropout=dropout, kernel_size=kernel_size, smooth_layer_num=smooth_layer)


def compute_val_loss(net, val_loader, criterion):
    """
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param epoch: int, current epoch
    :return: val_loss
    """

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            labels = labels.unsqueeze(-1)  # (B，N，T，1)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)

            # decode
            decoder_start_inputs = encoder_inputs[:, :, -1:, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)
            tmp.append(loss.item())
            if batch_index % 50 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        validation_loss = sum(tmp) / len(tmp)

    return validation_loss


def train():
    os.makedirs(params_path)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * 100, gamma=0.5)

    global_step = 0
    start_time = time()
    params_filename = os.path.join(params_path, 'netparamsCI.pth')
    step = []
    losslist = []
    valloss = []

    for epoch in range(epochs):
        net.train()
        train_start_time = time()
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            labels = labels.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = net(encoder_inputs, decoder_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            training_loss = loss.item()
            global_step += 1
            step = np.append(step, global_step)
            losslist = np.append(losslist, training_loss)
            print('epoch: %s, total time: %.2fs, train time every whole data: %.2fs, loss: %.4f' % (epoch + 1,
                                                                                                    time() - start_time,
                                                                                                    time() - train_start_time,
                                                                                                    training_loss),
                  flush=True)
        val_loss = compute_val_loss(net, val_loader, criterion)
        valloss = np.append(valloss, val_loss)
    torch.save(net.state_dict(), params_filename)
    np.savez('train.npz', step=step, trainloss=losslist, valloss=valloss)
