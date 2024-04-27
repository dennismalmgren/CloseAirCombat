import numpy as np
from .reward_function_base import BaseRewardFunction
from envs.JSBSim.core.catalog import Catalog as c

class SafeAltitudeReward(BaseRewardFunction):
    """
    SafeAltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mach
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        altitude = env.agents[agent_id].get_altitude_ground() / 1000    # unit: km
        ego_vz_neu = -env.agents[agent_id].get_velocity()[-1] / 340    # unit: mach
        Pv = 0.
        if altitude <= self.safe_altitude:
            Pv = -np.clip(ego_vz_neu / self.Kv * (self.safe_altitude - altitude) / self.safe_altitude, 0., 1.)
        PH = 0.
        if altitude <= self.danger_altitude:
            PH = np.clip(altitude / self.danger_altitude, 0., 1.) - 1. - 1.
        new_reward = Pv + PH
        return self._process(new_reward, agent_id, (Pv, PH))
