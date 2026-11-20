import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

class MLP_baseline(nn.Module):
    def __init__(self, num_features, embedding_dim, num_hidden, num_output):
        super(MLP_baseline, self).__init__()  
        # embedding layer
        self.embedding = nn.Embedding(num_output, embedding_dim)
        
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_features * embedding_dim), nonlinearity='relu'))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        
        # 1st hidden layer
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_hidden), nonlinearity='relu'))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # 2nd hidden layer
        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden), nonlinearity='linear'))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        self.norm3 = nn.BatchNorm1d(num_hidden)
        self.dropout3 = nn.Dropout(p=0.2)

        # define activation function
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # input layer
        x = self.embedding(x)
        x = x.flatten(start_dim=1)  # Flatten the embedding output to (batch_size, embedding_dim)
        
        # 1st hidden layer
        x = F.linear(x, self.W_1, self.b_1)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # 2nd hidden layer
        x = F.linear(x, self.W_2, self.b_2)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        
        # output layer
        x = F.linear(x, self.W_3, self.b_3)
        return x
