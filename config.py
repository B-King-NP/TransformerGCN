class Config:
    def __init__(self):
        self.adj_filename = 'CIadj.npy'
        self.matrix_file = 'CI.npz'
        self.num_for_predict = 10
        self.d_model = 64
        self.encoder_input_size = 1
        self.decoder_output_size = 1
        self.n_head = 8
        self.lr = 0.0001
        self.num_layers = 3
        self.epochs = 300
        self.smooth_layer_num = 1
        self.dropout = 0.2
        self.kernel_size = 3
        self.params_path = 'netparamsCI.pth'
        self.parapth = 'netparamsCI.pth'
        self.batch_size = 32
