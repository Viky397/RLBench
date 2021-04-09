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

def get_activation(name):
    return getattr(F, name) if name else lambda x: x


def init_(module):
    # could have different gains with different activations
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module

class MLP(nn.Module):
    """ MLP network (can be used as value or policy)
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[],
                 act="relu",
                 output_act=None,
                 init_weights=False,
                 **kwargs):
        """ multi-layer perception / fully-connected network

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dims (list): hidden layer dimensions
            act (str): hidden layer activation
            output_act (str): output layer activation
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        init_func = init_ if init_weights else lambda x: x

        self.fcs = nn.ModuleList([
            init_func(nn.Linear(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ])
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x):
        out = x
        for fc in self.fcs[:-1]:
            out = self.act(fc(out))
        out = self.output_act(self.fcs[-1](out))
        return out

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
        self.model =  MLP(40, 7, [128, 128, 128], act='tanh')
        # One-hot encodings
        self.target_enc = np.asarray([1, 0, 0])
        self.distract_enc = np.asarray([0, 1, 0])
        self.gripper_enc = np.asarray([0, 0, 1])

        checkpoint = torch.load("/home/mustafa/code/RLBench/trained_models/mlp_128tanh/mlp.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        dataset = []
        NUM_NODES = 4

        gripper_position = obs.gripper_pose[:3]
        relative_target_position = obs.task_low_dim_state[0][:3] - gripper_position
        relative_distractor0_position = obs.task_low_dim_state[1][:3] - gripper_position
        relative_distractor1_position = obs.task_low_dim_state[2][:3] - gripper_position

        target_node = np.concatenate(
            [relative_target_position, obs.task_low_dim_state[0][3:], self.target_enc])
        distract_node = np.concatenate(
            [relative_distractor0_position, obs.task_low_dim_state[1][3:], self.distract_enc])
        distract2_node = np.concatenate(
            [relative_distractor1_position, obs.task_low_dim_state[2][3:], self.gripper_enc])
        gripper_node = np.concatenate(
            [obs.gripper_pose, self.gripper_enc])

        # target_node = np.concatenate([obs.task_low_dim_state[0], self.target_enc])
        # distract_node = np.concatenate([obs.task_low_dim_state[1], self.distract_enc])
        # distract2_node = np.concatenate([obs.task_low_dim_state[2], self.distract_enc])
        # gripper_node = np.concatenate([obs.gripper_pose, self.gripper_enc])

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

        for data in loader:
            flat_x = data.x.reshape(-1, 40)            
            out = self.model(flat_x)
            break

        arm = out[0].detach().numpy() # Only one thing in batch

        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_size)

training_steps = 500
episode_length = 100
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)

    action = agent.act(obs)
    #print(obs.gripper_pose)
    #print(action)
    #print("action:", action)
    obs, reward, terminate = task.step(action)
    if (terminate):
        print('Success!')
        break

print('Done')
env.shutdown()
