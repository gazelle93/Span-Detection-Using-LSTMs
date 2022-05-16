import torch
import torch.nn as nn

class SpanDetectionLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout_rate, is_bilstm=False):
        super(SpanDetectionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm_layers = nn.ModuleList()

        if is_bilstm == False:
            for i in range(num_layers-1):
                self.lstm_layers.append(nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim))

            self.lstm_layers.append(nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim))
            self.ff_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

        else:
            for i in range(num_layers-1):
                self.lstm_layers.append(nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim//2, bidirectional=True))

            self.lstm_layers.append(nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim//2, bidirectional=True))
            self.ff_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, _input):
        for i in range(self.num_layers):
            output, _ = self.lstm_layers[i](_input)
            output = self.dropout(output)
        output = self.activation(output)
        
        output = self.ff_layer(output)

        return output
