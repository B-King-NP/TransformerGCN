import numpy as np
import torch


def normalization(x, max, min):
    x = (x - min) / (max - min)
    x = x * 2. - 1.
    return x


def re_normalization(x, max, min):
    x = (x + 1.) / 2.
    x = x * (max - min) + min
    return x


def load_graphdata(graph_signal_matrix_filename, DEVICE, batch_size, shuffle=True):
    """
    :param graph_signal_matrix_filename: str
    :param DEVICE:
    :param batch_size: int
    :param shuffle: bool
    :return:
    three DataLoaders, each dataloader contains:
    train_x_tensor: (B, N_nodes, in_feature, T_input)
    train_decoder_input_tensor: (B, N_nodes, T_output)
    train_target_tensor: (B, N_nodes, T_output)
    """

    file_data = np.load(graph_signal_matrix_filename)

    train_x = file_data['train_x']  # shape: (dataset_size, nodes, 1, input_length)
    train_target = file_data['train_target']  # shape: (dataset_size, nodes, prediction_length)

    val_x = file_data['val_x']
    val_target = file_data['val_target']

    max = file_data['max']  # (1, 1, 1, 1)
    min = file_data['min']  # (1, 1, 1, 1)

    # normalize
    train_x = normalization(train_x, max, min)
    train_target_norm = normalization(np.expand_dims(train_target, axis=2), max, min)
    train_target_norm = np.squeeze(train_target_norm, 2)

    val_x = normalization(val_x, max, min)
    val_target_norm = normalization(np.expand_dims(val_target, axis=2), max, min)
    val_target_norm = np.squeeze(val_target_norm, 2)

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, :, -1:]  # (B, N, 1(F), 1(T)), the last time step of the input is used as the initial input for the decoder
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_decoder_input_start = val_x[:, :, :, -1:]
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, train_target_tensor, val_loader, val_target_tensor, max, min
