import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import SafeAltitudeReward, OpusAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..utils.utils import LLA2NED, NED2LLA

class OpusAltitudeTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            OpusAltitudeReward(self.config),
            SafeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.position_h_sl_m,                  # 0. altitude  (unit: m)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.attitude_phi_rad,                 # 12. roll      (unit: rad)
            c.attitude_theta_rad,               # 13. pitch     (unit: rad)
            c.attitude_psi_rad,                 # 14. yaw     (unit: rad)
            c.attitude_heading_true_rad,        # 18. heading (unit: rad)
            c.velocities_vc_mps,                # 19. vc        (unit: m/s)
            c.position_h_agl_m                  # 22. altitude above ground level (unit: m)
        ]

        self.mission_var = [
            c.missions_cruise_target_position_h_sl_m,
        ]

        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(13,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(4,))

    def reset(self, env):
        super().reset(env)
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            agent.set_property_value(c.missions_cruise_target_position_h_sl_m, current_altitude)
    
    def calculate_mission_variables(self, env, agent_id):
        #construct the heading task observation.
        target_altitude = np.asarray([self.mission_props[0]])
        #convert to relative values.
        delta_altitude = target_altitude - self.state_props[0]
        return delta_altitude
    
    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.
        observation space:
        0. current task id
        1. task 1 type id
        2. task 2 type id
        """
        agent = env.agents[agent_id]
        self.state_props = np.array(agent.get_property_values(self.state_var))
        self.ned_frame_state = self.convert_state_to_ned_frame(self.state_props)
        # self.action_props = np.array(agent.get_property_values(self.action_var))
        self.mission_vars = np.array(agent.get_property_values(self.mission_var))

        delta_altitude, delta_heading, delta_speed = self.calculate_heading_task_variables(env, agent_id)

        
        heading_mission_vars = np.concatenate([delta_altitude, delta_heading, delta_speed])
        heading_mission_vars[0] /= 5000 #delta altitude (unit: 5km)
        heading_mission_vars[3] /= 340 #delta speed (unit: mach)
        uvw = self.ned_frame_state[6:9]
        uvw /= 340 #unit: mach
        attitude = self.ned_frame_state[12:16]
        altitude = self.state_props[0:1]
        altitude /= 5000 #unit: 5km
        speed_vc = self.state_props[19:20]
        speed_vc /= 340 #unit: mach
        obs = np.concatenate([heading_mission_vars, uvw, attitude, altitude, speed_vc])

        norm_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        return action
