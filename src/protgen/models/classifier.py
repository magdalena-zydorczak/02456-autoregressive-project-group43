import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

class ProteinFamilyClassifier(nn.Module):
    def __init__(self, num_features, vocab_size, embedding_dim, num_hidden, num_output):
        super(ProteinFamilyClassifier, self).__init__()  
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_features * embedding_dim), nonlinearity='relu'))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        
        # Hidden layer
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden), nonlinearity='linear'))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        self.norm2 = nn.LayerNorm(num_hidden)
        self.dropout2 = nn.Dropout(p=0.2)

        # define activation function
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # input layer
        x = self.embedding(x)
        x = x.flatten(start_dim=1)  # Flatten the embedding output to (batch_size, embedding_dim)
        
        # Hidden layer
        x = F.linear(x, self.W_1, self.b_1)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # output layer
        x = F.linear(x, self.W_2, self.b_2)
        return x