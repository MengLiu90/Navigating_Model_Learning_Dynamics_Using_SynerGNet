import torch
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

class Net(torch.nn.Module):
   def __init__(self, num_layers, num_node_features, num_classes):
      super(Net, self).__init__()

      self.conv1 = SAGEConv(num_node_features, 512)
      self.batch_norm1 = BatchNorm1d(512)
      self.fc1 = Linear(2 * 512, 512)
      self.batch_norm4 = BatchNorm1d(512)
      self.dropout = Dropout(0.65)
      self.fc2 = Linear(512, num_classes)


   def forward(self, data):
      x, edge_index, batch = data.x, data.edge_index, data.batch

      x = self.conv1(x, edge_index)
      x = self.batch_norm1(x)
      x = F.relu(x)
      x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
      x = self.fc1(x)
      x = self.batch_norm4(x)
      x = F.relu(x)
      x = self.dropout(x)
      logits = self.fc2(x)
      return logits

   def reset_parameters(self):
      for module in self.modules():
         if isinstance(module, (Linear, SAGEConv, BatchNorm1d)):
            module.reset_parameters()