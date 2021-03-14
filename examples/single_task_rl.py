from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size
        self.model =  SimpleGCNModel(6, 3)
        # One-hot encodings
        self.target_enc = np.asarray([1, 0, 0])
        self.distract_enc = np.asarray([0, 1, 0])
        self.gripper_enc = np.asarray([0, 0, 1])

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        obs.task_low_dim_state[0]
        
        target_node = np.concatenate((data[k][0], self.target_enc))
        distract_node = np.concatenate((data[k][1], self.distract_enc))
        distract2_node = np.concatenate((data[k][2], self.distract_enc))
        gripper_node = np.concatenate((data[k][3], self.gripper_enc))
        nodes = torch.tensor([target_node, distract_node, distract2_node, gripper_node], dtype=torch.float)

        edge_index = torch.tensor([[0, 1],
                                           [1, 0],
                                           [0, 2],
                                           [2, 0],
                                           [0, 3],
                                           [3, 0],
                                           [1, 2],
                                           [2, 1],
                                           [1, 3],
                                           [3, 1],
                                           [2, 3],
                                           [3, 2]], dtype=torch.long)
        y = torch.tensor([data[k + 1][3]], dtype=torch.float)
        graph_data = Data(x=nodes, edge_index=edge_index.t().contiguous(), y=y)
        
        out = model(graph_data.x, graph_data.edge_index, 1)

        arm = out[0] # Only one thing in batch
        # gripper = [1.0]  # Always open
        rotation = [0, 0, 0]

        return np.concatenate([arm, rotation], axis=-1)

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_WORLD_FRAME)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_size)

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
