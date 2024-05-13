import numpy as np
from gymnasium import spaces
from collections import deque

from .task_base import BaseTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..reward_functions.cruise_missile_event_driven_reward import CruiseMissileEventDrivenReward
from ..reward_functions.cruise_missile_posture_reward import CruiseMissilePostureReward
from ..core.simulator import MissileSimulator
from ..core.cruise_missile_simulator import CruiseMissileSimulator, AntiCruiseMissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
from ..termination_conditions.cruise_missile_success import CruiseMissileSuccess
from ..core.catalog import Catalog as c
import torch

from tensordict import TensorDict

class SingleMissileDefenseTask(BaseTask):
    """This task aims at training agent to engage incoming cruise missiles
    """
    def __init__(self, config, lowlevelpolicy):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 90) #180
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', 15000) #20 km
        #self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf) #todo, try with 5 km
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)

        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])
        self.lowlevelpolicy = lowlevelpolicy

        self.reward_functions = [
            AltitudeReward(self.config),
            CruiseMissilePostureReward(self.config),
            CruiseMissileEventDrivenReward(self.config)
        ]


        self.termination_conditions = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            CruiseMissileSuccess(self.config)
        ]

        self.truncation_condition = Timeout(self.config)

        self.missile = None

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    @property
    def num_agents(self):
        return 1
    
    def load_action_space(self):
        self.action_space = spaces.Discrete(3 * 5 * 3)
        self.action_dims = [3, 5, 3]
    
    def normalize_action(self, env, agent_id, action):
        high_level_action = self.convert_to_multi_discrete(action)
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[high_level_action[0]]
        input_obs[1] = self.norm_delta_heading[high_level_action[1]]
        input_obs[2] = self.norm_delta_velocity[high_level_action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        input_obs = torch.tensor(input_obs, dtype=torch.float32)
        input_data = {
            "observation": input_obs
        }
        input_td = TensorDict(source=input_data, batch_size=[]).to(self.lowlevelpolicy.device)
        input_td = self.lowlevelpolicy(input_td)
        low_level_action = input_td["action"].cpu().numpy().squeeze()
        # if low_level_action.ndim == 2:
        #     low_level_action[0, -1] = np.clip(low_level_action[0, -1], 0.0, 0.9) #TODO: Fix hacklow_level_action[0]
        # else:
        #     low_level_action[-1] = np.clip(low_level_action[-1], 0.0, 0.9)
        return low_level_action
    
    def convert_to_multi_discrete(self, single_action):
        multi_discrete_action = []
        for dim in self.action_dims:
            action = single_action % dim
            multi_discrete_action.append(action)
            single_action = single_action // dim
        return np.asarray(multi_discrete_action[::-1], dtype=np.int32) # reverse the order

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    #just train on a single missile.
    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative missile info
            - [9] delta_v_body_x
            - [10] delta altitude
            - [11] ego_AO
            - [12] ego_TA
            - [13] relative distance
            - [14] side flag
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4]) 
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        if self.missile is not None and self.missile.is_alive:
            ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
            missile_feature = np.concatenate((self.missile.get_position(), self.missile.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[9] = (np.linalg.norm(self.missile.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[10] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[11] = ego_AO
            norm_obs[12] = ego_TA
            norm_obs[13] = R / 10000
            norm_obs[14] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_anticruise_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
          #at what altitude? lets say single point of origin for now.
        missile_start_lat =  np.random.choice([60.1, 59.5]).item()
        missile_start_lon = 120.3
        missile_start_altitude = 4000 #m
        #missile_heading = 180 #deg
        missile_speed = 0.6 * 340 # ft/s
        start_geodetic = np.asarray([missile_start_lon, missile_start_lat, missile_start_altitude])
        missile_tgt_lat = np.random.choice([60.1, 59.7]).item()
        missile_tgt_lon = missile_start_lon - 1.0
        missile_tgt_altitude = 400 / 3.0 #m 
        tgt_geodetic = np.asarray([missile_tgt_lon, missile_tgt_lat, missile_tgt_altitude])
        origin = env.center_lon, env.center_lat, env.center_alt
        uid = "RM01"
        dt = 1 / 60.0
        env._tempsims.clear()
        self.missile = CruiseMissileSimulator.create(start_geodetic, 
                                                tgt_geodetic, 
                                                origin,
                                                uid, 
                                                missile_speed,
                                                dt)
        env.add_temp_simulator(self.missile)
        return super().reset(env)
      
    def step(self, env):
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target = self.missile.get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    AntiCruiseMissileSimulator.create(parent=agent, target=self.missile, uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step

