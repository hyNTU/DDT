import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.init import xavier_normal_



class CDAE(nn.Module):
    def __init__(self, num_users: int, num_items: int, hidden_dim: int, corruption_ratio: float):
        super(CDAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.corruption_ratio = corruption_ratio
        self.drop_out = nn.Dropout(p=corruption_ratio)

        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.encoder = nn.Linear(num_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_items)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)

    def forward(self, user_idx, x):
        # apply corruption
        x_corrupted = self.drop_out(x)
        encoder = self.relu(self.encoder(x_corrupted) + self.user_embedding(user_idx))
        return self.decoder(encoder)

class CombinedNetwork(nn.Module):
    def __init__(self, user_num, item_num, factor_num,
                    dropout, num_models):
        super(CombinedNetwork, self).__init__()
        self.num_models = num_models
        self.combined_network = nn.ModuleList([])
        for _ in range(num_models):
            self.combined_network.append(CDAE(user_num, item_num, factor_num, dropout))

    def forward(self, user, data, model_id):
        
        return(self.combined_network[model_id](user, data))



class SelfAttention(nn.Module):
    def __init__(self, num_models, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Embedding(num_models, dim) # [5 x dim]
        self.key = nn.Embedding(num_models, dim) # [5 x dim]
        # self.query = nn.Parameter(torch.ones(num_models, dim)) # [5 x dim]
        # self.key = nn.Parameter(torch.ones(num_models, dim)) # [5 x dim]
        self.softmax = nn.Softmax(dim=-1)
        self.mask = torch.eye(num_models).bool().cuda()

        self._init_weight_(self)

    def _init_weight_(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def temperature_scaled_softmax(self, logits, temperature):
        logits = logits / temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, x, group):
        # x shape: (5, 2048)
        # Transpose to (2048, 5) for batch processing
        x = x.transpose(0, 1)  # Shape now (2048, 5)
                
        # Compute attention scores
        scores = torch.matmul(self.key.weight,self.query.weight.transpose(0,1)) / (self.query.weight.size(-1) ** 0.5)

        # scores = torch.matmul(self.query,self.key.transpose(0,1)) / (self.key.size(-1) ** 0.5)
        scores = scores.masked_fill(self.mask, -float('Inf'))
        # attention = self.softmax(scores)
        attention = self.temperature_scaled_softmax(scores, torch.tensor(3.0).cuda())
        # Apply attention to values
        out = x * attention[group] # Shape (2048, 5)
        
        return torch.sum(out, dim=-1)  # This is (2048, 1) 

    def massforward(self, x, group):
        # x shape: (5, 2048)
        # Transpose to (2048, 5) for batch processing
        x = x.transpose(0, 1)  # Shape now (2048, 5)
                
        # Compute attention scores
        scores = torch.matmul(self.key.weight,self.query.weight.transpose(0,1)) / (self.query.weight.size(-1) ** 0.5)

        # scores = torch.matmul(self.query,self.key.transpose(0,1)) / (self.key.size(-1) ** 0.5)
        scores = scores.masked_fill(self.mask, -float('Inf'))
        # attention = self.softmax(scores)
        attention = self.temperature_scaled_softmax(scores, torch.tensor(3.0).cuda())
        
        # Apply attention to values
        out = x * attention[group].unsqueeze(-1) # Shape (2048, 5, item_num)
        
        return torch.sum(out, dim=1)  # This is (2048, 1, item_num) 