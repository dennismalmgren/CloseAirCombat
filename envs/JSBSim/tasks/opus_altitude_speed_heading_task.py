import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import (
    SafeAltitudeReward, 
    OpusAltitudeSpeedHeadingReward
)
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..utils.utils import LLA2NED, NED2LLA

class OpusAltitudeSpeedHeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        #self.lat0, self.lon0, self.alt0 = config.battle_field_origin
        #self.n0, self.e0, self.u0 = 0, 0, 0

        self.reward_functions = [
            OpusAltitudeSpeedHeadingReward(self.config),
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
        self.state_props = [
            c.position_h_sl_m,                  # 0. altitude  (unit: m)
            c.velocities_u_mps,                 # 1. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 2. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 3. v_body_z   (unit: m/s)
            c.attitude_phi_rad,                 # 4. roll      (unit: rad)
            c.attitude_theta_rad,               # 5. pitch     (unit: rad)
            c.attitude_psi_rad,                 # 6. yaw     (unit: rad)
            c.velocities_vc_mps,                # 7. vc        (unit: m/s)
            c.attitude_heading_true_rad,         # 8. heading   (unit: rad)
        ]
        
        self.mission_props = [
            c.missions_cruise_target_position_h_sl_m,
            c.missions_cruise_target_attitude_heading_true_rad,
            c.missions_cruise_target_velocities_vc_mps,
        ]

        self.action_props = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.0, 1.0]
        ]

    def load_observation_space(self):
        task_variable_count = 6
        state_variable_count = 11
        self.observation_space = spaces.Box(low=-10, high=10., shape=(task_variable_count + state_variable_count,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=np.asarray([-1.0, -1.0, -1.0, 0.0]),
                                       high=np.asarray([1.0, 1.0, 1.0, 0.9]), dtype=np.float32, shape=(4,))

    def reset(self, env):
        super().reset(env)
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_heading_rad = agent.get_property_value(c.attitude_heading_true_rad) 
            current_speed = agent.get_property_value(c.velocities_vc_mps) 

            agent.set_property_value(c.missions_cruise_target_position_h_sl_m, current_altitude)
            agent.set_property_value(c.missions_cruise_target_attitude_heading_true_rad, current_heading_rad)
            agent.set_property_value(c.missions_cruise_target_velocities_vc_mps, current_speed)

    
    def calculate_task_variables(self, env, agent_id):
        agent = env.agents[agent_id]
        #construct the heading task observation.
        target_altitude = self.mission_prop_vals[0]
        current_altitude = self.state_prop_vals[0]
        #convert to relative values.
        delta_altitude = target_altitude - current_altitude

        target_roll = 0.0
        current_roll = self.state_prop_vals[4]
        delta_roll = target_roll - current_roll        

        target_speed = self.mission_prop_vals[2]
        current_speed = self.state_prop_vals[7]
        delta_speed = target_speed - current_speed

        target_heading = self.mission_prop_vals[1]
        current_heading = self.state_prop_vals[8]
        delta_heading = target_heading - current_heading
        return np.asarray([delta_altitude, delta_roll, delta_speed, delta_heading])
    
    def get_obs(self, env, agent_id):
        agent = env.agents[agent_id]
        self.state_prop_vals = np.array(agent.get_property_values(self.state_props))
        self.mission_prop_vals = np.array(agent.get_property_values(self.mission_props))

        task_variables = self.calculate_task_variables(env, agent_id)
        task_variables = self.transform_task_variables(task_variables)        
        uvw = self.state_prop_vals[1:4].copy()
        uvw = self.transform_uvw(uvw)
        attitude = self.state_prop_vals[4:7].copy()
        attitude = self._convert_to_quaternion(attitude[0], attitude[1], attitude[2])
        attitude_heading = self.state_prop_vals[8]
        attitude_heading = self._convert_to_sincos(attitude_heading)
        speed_vc = self.state_prop_vals[7:8].copy()
        speed_vc /= 340 #unit: mach
        altitude = self.state_prop_vals[0:1].copy()
        altitude /= 5000 #unit: 5km
        obs = np.concatenate([uvw, attitude, attitude_heading, speed_vc, altitude, task_variables])

        norm_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def transform_uvw(self, uvw):
        return uvw / 340 #unit: mach
    
    def transform_task_variables(self, task_variables):
        task_variables[0] = task_variables[0] / 5000 #delta altitude (unit: 1km)
        task_variables[2] = task_variables[2] / 340  #delta velocity (unit: mach)

        task_variables = np.asarray([task_variables[0], 
                                     np.sin(task_variables[1]), 
                                     np.cos(task_variables[1]), 
                                     task_variables[2], 
                                     np.sin(task_variables[3]),
                                     np.cos(task_variables[3])])        
        return task_variables
    
    def normalize_action(self, env, agent_id, action):
        return action
