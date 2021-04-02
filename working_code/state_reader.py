import numpy as np
from pyquaternion import Quaternion

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

data = np.load('/home/mustafa/Desktop/reach_target/variation0/episodes/episode0/state_data.npy')
print('State Data:')
print(f'Shape: {data.shape}')
print(data[35][3])

q1x, q1y, q1z, q1w = data[35][3][3:]
q2x, q2y, q2z, q2w = data[36][3][3:]

q1 = Quaternion(q1w, q1x, q1y, q1z)
q2 = Quaternion(q2w, q2x, q2y, q2z)

print(f'Quaternion 1: {q1}')
print(f'Quaternion 2: {q2}')

delta_rot = q2 * q1.inverse
qw, qx, qy, qz = list(delta_rot)
            
x, y, z = data[36][3][:3] - data[35][3][:3]

diff = [x, y, z] + [qx, qy, qz, qw]

print(f'Delta: {diff}')

print(f'Applying Transform:')

a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = diff
x, y, z, qx, qy, qz, qw = data[35][3]
new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
qw, qx, qy, qz = list(new_rot)
new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]

print(f'New Pose: {new_pose}')
print(f'Actual Pose: {data[36][3]}')

class SimpleGCNModel(nn.Module):

    def __init__(self, num_node_features, num_output_channels):
        super(SimpleGCNModel, self).__init__()
        hidden_layers = 64
        self.conv1 = GCNConv(num_node_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, hidden_layers)
        self.conv3 = GCNConv(hidden_layers, hidden_layers)
        self.lin = nn.Linear(hidden_layers, num_output_channels)

    def forward(self, x, edge_index, batch):

        # Node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Node aggregation
        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

class Agent(object):

    def __init__(self):
        self.model =  SimpleGCNModel(10, 7)
        # One-hot encodings
        self.target_enc = np.asarray([1, 0, 0])
        self.distract_enc = np.asarray([0, 1, 0])
        self.gripper_enc = np.asarray([0, 0, 1])

        checkpoint = torch.load("/home/mustafa/code/RLBench/trained_models/gcn_apr01/graph_model.pth")
        print('Checkpoint Loaded')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def act(self, target, dis1, dis2, gripper):
        dataset = []
        NUM_NODES = 4

        target_node = np.concatenate([target, self.target_enc])
        distract_node = np.concatenate([dis1, self.distract_enc])
        distract2_node = np.concatenate([dis2, self.distract_enc])
        gripper_node = np.concatenate([gripper, self.gripper_enc])

        nodes = torch.tensor(
                [target_node, distract_node, distract2_node, gripper_node],
                dtype=torch.float)
        
        # Build edge relationships (Fully Connected)
        edge_index = torch.tensor([[i, j]
                                    for i in range(NUM_NODES)
                                    for j in range(NUM_NODES)
                                    if i != j],
                                    dtype=torch.long)

        graph_data = Data(x=nodes, edge_index=edge_index.t().contiguous())

        dataset.append(graph_data)
        loader = DataLoader(dataset, batch_size=1)

        for dat in loader:            
            out = self.model(dat.x, dat.edge_index, dat.batch)
            break

        arm = out[0].detach().numpy() # Only one thing in batch

        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

agent = Agent()
action = agent.act(data[35][4], data[35][5], data[35][6], data[35][3])
print(action)

