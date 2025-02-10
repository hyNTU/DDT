import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.init import xavier_normal_



class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                    dropout, model):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """		
        self.dropout = dropout
        self.model = model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """

        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1) # stack by to back
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


class CombinedNetwork(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                    dropout, model, num_models):
        super(CombinedNetwork, self).__init__()
        self.num_models = num_models
        self.combined_network = nn.ModuleList([])
        for _ in range(num_models):
            self.combined_network.append(NCF(user_num, item_num, factor_num, num_layers, dropout, model))

    def forward(self, user, item, model_id):
        
        return(self.combined_network[model_id](user, item))



class SelfAttention(nn.Module):
    def __init__(self, num_models, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Parameter(torch.ones(num_models, dim)) # [5 x dim]
        self.key = nn.Parameter(torch.ones(num_models, dim)) # [5 x dim]
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
        scores = torch.matmul(self.query,self.key.transpose(0,1))
        #  / (self.key.size(-1) ** 0.5)
        scores = scores.masked_fill(self.mask, -float('Inf'))
        # attention = self.softmax(scores)
        attention = self.temperature_scaled_softmax(scores, torch.tensor(3.0).cuda())
        
        # Apply attention to values
        out = x * attention[group] # Shape (2048, 5)
        
        return torch.sum(out, dim=-1)  # This is (2048, 1) 


