import numpy as np
import torch
import torch.utils.data

from config import Config
from model import make_model
from utils import re_normalization, load_graphdata

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
graph_signal_matrix_filename = config.matrix_file
params_path = config.params_path
parapth = config.parapth

train_loader, train_target_tensor, max, min = load_graphdata(graph_signal_matrix_filename, DEVICE, batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, n_head, num_for_predict,
                 dropout=dropout, kernel_size=kernel_size, smooth_layer_num=smooth_layer)


def predict(net, data_loader, data_target_tensor, max, min, parampth):
    net.load_state_dict(torch.load(parampth, map_location=torch.device('cpu')))
    net.train(False)
    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        prediction = []
        input = []
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            labels = labels.unsqueeze(-1)  # (B, N, T, 1)
            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, N, T, F(1))

            # decode
            decoder_start_inputs = encoder_inputs[:, :, -1:, :]  # the last time step of the input is used as the initial input for the decoder
            decoder_input_list = [decoder_start_inputs]

            # predicting step by step
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())

        input = np.concatenate(input, 0)
        input = np.squeeze(input, 3)
        input = np.expand_dims(input, 2)
        input = re_normalization(input, max, min)
        input = np.squeeze(input, 2)

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = np.squeeze(prediction, 3)
        prediction = np.expand_dims(prediction, 2)
        prediction = re_normalization(prediction, max, min)
        prediction = np.squeeze(prediction, 2)

        data_target_tensor = np.expand_dims(data_target_tensor, 2)
        data_target_tensor = re_normalization(data_target_tensor, max, min)
        data_target_tensor = np.squeeze(data_target_tensor, 2)  # (batch, N, T')

        return input, prediction, data_target_tensor
