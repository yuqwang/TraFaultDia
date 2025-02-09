import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):

    def __init__(self, config):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()

        self.config = config

        for i, (name, param) in enumerate(self.config):
            if name == 'rnn':
                """
                  :param config: [input_size, hidden_size, n_layers, output_size]
                  [Nshots, max_tokens, embedding_length]
                        """
                self.input_size = param[0]
                self.hidden_size = param[1]
                self.n_layers = param[2]
                self.output_size = param[3]
                self.dropout_rate = param[4]

                if self.n_layers > 1:
                    self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers,
                                      batch_first=True, dropout=self.dropout_rate)
                else:
                    self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first=True)

                self.linear = nn.Linear(self.hidden_size, self.output_size)

                #self.hidden1 = None


            elif name == 'lstm':
                print("lstm")
                """
                                  :param config: [input_size, hidden_size, n_layers, output_size]
                                  [Nshots, max_tokens, embedding_length]
                                        """
                self.input_size = param[0]
                self.hidden_size = param[1]
                self.n_layers = param[2]
                self.output_size = param[3] # Set your dropout rate
                self.dropout_rate = param[4]

                self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
                self.linear = nn.Linear(self.hidden_size, self.output_size)
                self.hidden2 = None

            elif name == 'cnn':
                print("cnn")
                """
                Assuming param for CNN: [feature_dim, n_filters, filter_sizes, output_size, dropout]
                """
                feature_dim = param[0]
                n_filters = param[1]
                filter_sizes = param[2]
                output_size = param[3]
                dropout = param[4]

                self.convs = nn.ModuleList([
                    nn.Conv1d(in_channels=feature_dim, out_channels=n_filters, kernel_size=fs)
                    for fs in filter_sizes
                ])

                self.dropout = nn.Dropout(dropout)
                # The number of input features to the linear layer is n_filters times the number of filter_sizes
                self.fc = nn.Linear(n_filters * len(filter_sizes), output_size)

            elif name == 'transformer_encoder':
                #print("transformer_encoder is initialized")

                """
                :param config: [nhead, num_encoder_layers, num_decoder_layers, d_model, dim_feedforward]
                """
                self.d_model = param[0]
                self.nhead = param[1]
                self.num_encoder_layers = param[2]
                #self.num_decoder_layers = param[2]
                self.dim_feedforward = param[3]
                self.output_size = param[4]
                self.dropout_rate = param[5]

                # Define transformer encoder layer
                encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                            nhead=self.nhead,
                                                            dim_feedforward=self.dim_feedforward,
                                                            dropout=self.dropout_rate,
                                                            batch_first=True)

                # Define transformer encoder
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                                 num_layers=self.num_encoder_layers)

                self.linear = nn.Linear(self.d_model, self.output_size)

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue

            else:
                raise NotImplementedError

    def forward(self, x):
        """
                This function can be called by forward & finetunning from MAML.
                :param x: [setsz, max_tokens, embedding_length]
                :return: x, loss, likelihood, kld
                """
        for name, param in self.config:
            if name == 'rnn':
                self.rnn.flatten_parameters()  
                h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

                output, _ = self.rnn(x, h0)
                output = self.linear(output[:, -1, :])

            elif name == 'cnn':
                # CNN forward logic
                # Need to permute x to match Conv1d input requirements
                x = x.permute(0, 2, 1)
                x = [F.relu(conv(x)) for conv in self.convs]
                x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
                x = self.dropout(torch.cat(x, dim=1))
                output = self.fc(x)

            elif name == 'transformer_encoder':
                #print("transformer_encoder forward")
                #src = x.permute(1, 0, 2)  # Switch to [seq_len, batch, features]
                output = self.transformer_encoder(x)
                output = output[:, -1, :]
                output = self.linear(output)  # Aggregate the output (you might use mean, max, etc.)

            elif name == 'lstm':
                setsz = x.size(0)

                if self.hidden2 is None or self.hidden2[0].size(1) != setsz:
                    self.init_hidden(setsz, x.is_cuda, mode="lstm")

                output, hidden = self.lstm(x, self.hidden2)

                output = output[:, -1, :]

                output = self.linear(output)

        return output

    def updated_forward(self, x, fast_weights):
        for p, new_p in zip(self.parameters(), fast_weights):
            p.data = new_p.data
        output = self.forward(x)
        return output

    def init_hidden(self, input_setsz, gpu = True, mode ="rnn"):
        weight = next(self.parameters()).data
        if mode == 'rnn':
            if (gpu):
                self.hidden1 = weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda()
            else:
                self.hidden1 =weight.new(self.n_layers, input_setsz, self.hidden_size).zero_()

        elif mode == 'lstm':
            if (gpu):
                self.hidden2 = (weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda(),
                                weight.new(self.n_layers, input_setsz, self.hidden_size).zero_().cuda())
            else:
                self.hidden2 = (weight.new(self.n_layers, input_setsz, self.hidden_size).zero_(),
                                weight.new(self.n_layers, input_setsz, self.hidden_size).zero_())

