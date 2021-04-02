from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


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

    def __init__(self, action_size):
        self.action_size = action_size
        self.model =  SimpleGCNModel(10, 7)
        # One-hot encodings
        self.target_enc = np.asarray([1, 0, 0])
        self.distract_enc = np.asarray([0, 1, 0])
        self.gripper_enc = np.asarray([0, 0, 1])

        checkpoint = torch.load("/home/mustafa/code/RLBench/trained_models/gcn_apr01/graph_model.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        dataset = []
        NUM_NODES = 4

        target_node = np.concatenate([obs.task_low_dim_state[0], self.target_enc])
        distract_node = np.concatenate([obs.task_low_dim_state[1], self.distract_enc])
        distract2_node = np.concatenate([obs.task_low_dim_state[2], self.distract_enc])
        gripper_node = np.concatenate([obs.gripper_pose, self.gripper_enc])

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

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
env = Environment(action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_size)

training_steps = 120
episode_length = 100
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)

    action = agent.act(obs)
    print(obs.gripper_pose)
    print(action)
    #print("action:", action)
    obs, reward, terminate = task.step(action)
    if (terminate):
        print('Success!')
        break

print('Done')
env.shutdown()
