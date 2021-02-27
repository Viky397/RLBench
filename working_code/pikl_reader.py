import pickle 

# pikl_file =# pikl_file = "/home/veronica/Downloads/rlbench_data/reach_target/variation0/episodes/episode0/low_dim_obs.pkl"
pikl_file = "/tmp/rlbench_data/reach_target/variation0/episodes/episode0/low_dim_obs.pkl"

data = pickle.load( open(pikl_file, "rb" ) )

for obs in data.__dict__['_observations']:
    vel = obs.joint_velocities
    state = obs.task_low_dim_state

import pdb;pdb.set_trace()
print(data)