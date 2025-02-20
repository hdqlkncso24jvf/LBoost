import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        self.conv_1 = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(in_channels + hidden_channels, out_channels)
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    def forward(self, x, edge_index):
        x_1 = self.conv_1(x, edge_index)
        x_2 = self.conv_2(x_1, edge_index)
        x_3 = self.fc(torch.cat([x, x_2], dim=1))
        return F.log_softmax(x_3, dim=1)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv_1 = GATConv(in_channels, hidden_channels)
        self.conv_2 = GATConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(in_channels + hidden_channels, out_channels)
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    def forward(self, x, edge_index):
        x_1 = self.conv_1(x, edge_index)
        x_2 = self.conv_2(x_1, edge_index)
        x_3 = self.fc(torch.cat([x, x_2], dim=1))
        return F.log_softmax(x_3, dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv_1 = SAGEConv(in_channels, hidden_channels)
        self.conv_2 = SAGEConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(in_channels + hidden_channels, out_channels)
        nn.init.uniform_(self.fc.weight, a=-1, b=1)

    def forward(self, x, edge_index):
        x_1 = self.conv_1(x, edge_index)
        x_2 = self.conv_2(x_1, edge_index)
        x_3 = self.fc(torch.cat([x, x_2], dim=1))
        return F.log_softmax(x_3, dim=1)
