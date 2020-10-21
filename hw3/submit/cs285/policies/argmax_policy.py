import numpy as np
import cs285.infrastructure.pytorch_util as ptu
import torch


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        obs = ptu.from_numpy(obs)
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output

        actions = self.critic.q_net(obs)

        return torch.argmax(actions.squeeze())
