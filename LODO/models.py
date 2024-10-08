from library_imports import *
from config import device, writer
# from main import *


################################################################################

# ----------------------------- Configuration 3 ---------------------------------
# -----------------------------                  --------------------------------
class config3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(config3, self).__init__()

        self.conv1 = tg_nn.GraphConv(input_dim, hidden_dim, aggr='add')
        self.conv2 = tg_nn.GraphConv(hidden_dim, hidden_dim, aggr='add')
        self.conv3 = tg_nn.GraphConv(hidden_dim, hidden_dim, aggr='max')
        self.conv4 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv5 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv6 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')

        self.jk1 = JumpingKnowledge("lstm", hidden_dim, 3)
        self.jk2 = JumpingKnowledge("lstm", hidden_dim, 3)

        self.lin1 = torch.nn.Linear(hidden_dim, 63)
        self.lin2 = torch.nn.Linear(63, 3)

        self.active1 = nn.PReLU(hidden_dim)
        self.active2 = nn.PReLU(hidden_dim)
        self.active3 = nn.PReLU(hidden_dim)
        self.active4 = nn.PReLU(hidden_dim)
        self.active5 = nn.PReLU(hidden_dim)
        self.active6 = nn.PReLU(hidden_dim)
        self.active7 = nn.PReLU(63)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_weight = 1 / edge_weight
        edge_weight = edge_weight.float()

        x = self.conv1(x, edge_index, edge_weight)
        x = self.active1(x)
        xs = [x]

        x = self.conv2(x, edge_index, edge_weight)
        x = self.active2(x)
        xs += [x]

        x = self.conv3(x, edge_index, edge_weight)
        x = self.active3(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk1(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.conv4(x, edge_index)
        x = self.active4(x)
        xs = [x]

        x = self.conv5(x, edge_index)
        x = self.active5(x)
        xs += [x]

        x = self.conv6(x, edge_index)
        x = self.active6(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk2(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.lin1(x)
        x = self.active7(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)

        return x

    def loss(self, pred, label):
        return (torch.sqrt(
            ((pred[:, 0] - label[:, 0]) ** 2).unsqueeze(-1) + ((pred[:, 1] - label[:, 1]) ** 2).unsqueeze(-1) + (
                        (pred[:, 2] - label[:, 2]) ** 2).unsqueeze(-1))).sum()

        # return (pred - label).abs().sum()  #MAE
        # return F.mse_loss(pred, label)  #MSE

################################################################################