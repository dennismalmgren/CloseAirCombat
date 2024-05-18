import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import (
    SafeAltitudeReward, 
    OpusAltitudeSpeedHeadingReward,
    OpusAltitudeReward,
    OpusSmoothingReward
)
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..utils.utils import LLA2NED, NED2LLA

class OpusSmoothingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        #self.lat0, self.lon0, self.alt0 = config.battle_field_origin
        #self.n0, self.e0, self.u0 = 0, 0, 0

        self.reward_functions = [
          #  OpusAltitudeSpeedHeadingReward(self.config),
            OpusSmoothingReward(self.config),
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
            c.attitude_phi_rad,                
            c.attitude_theta_rad,            
            c.attitude_psi_rad,                 
            c.velocities_u_mps,                
            c.velocities_v_mps,                
            c.velocities_w_mps,                
            c.velocities_vc_mps,               
            c.velocities_p_rad_sec,           
            c.velocities_q_rad_sec,             
            c.velocities_r_rad_sec,            
            c.accelerations_udot_m_sec2,
            c.accelerations_vdot_m_sec2,
            c.accelerations_wdot_m_sec2,
            c.accelerations_pdot_rad_sec2,
            c.accelerations_qdot_rad_sec2,
            c.accelerations_rdot_rad_sec2,
            
            #todo: consider these..
            c.attitude_heading_true_rad,         
            c.position_h_sl_m,               
#            c.velocities_h_dot_mps
        ]
        
        self.mission_props = [
            c.missions_cruise_target_position_h_sl_m,
            c.missions_cruise_target_velocities_vc_mps,
            c.missions_cruise_target_attitude_heading_true_rad,
        ]

        self.action_props = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.0, 1.0]
        ]

    def load_observation_space(self):
        task_variable_count = 4
        state_variable_count = 18 + 4
        action_variable_count = 4
        self.observation_space = spaces.Box(low=-100, high=100., shape=(task_variable_count + state_variable_count + action_variable_count,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=np.asarray([-1.0, -1.0, -1.0, 0.0]),
                                       high=np.asarray([1.0, 1.0, 1.0, 0.9]), dtype=np.float32, shape=(4,))

    def reset(self, env):
        super().reset(env)
        self.step_num = 1
        self.reset_task_history(env)
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            current_altitude = agent.get_property_value(c.position_h_sl_m)
            current_heading_rad = agent.get_property_value(c.attitude_heading_true_rad) 
            current_speed = agent.get_property_value(c.velocities_vc_mps) 

            agent.set_property_value(c.missions_cruise_target_position_h_sl_m, current_altitude)
            agent.set_property_value(c.missions_cruise_target_attitude_heading_true_rad, current_heading_rad)
            agent.set_property_value(c.missions_cruise_target_velocities_vc_mps, current_speed)

    def step(self, env):
        super().step(env)
        self.step_task_history(env)
        self.step_num += 1

    def reset_task_history(self, env):
        self.p_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.q_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.r_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.pdot_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.qdot_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.rdot_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.step_task_history(env)

    def step_task_history(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.p_history[agent_id][:4] = self.p_history[agent_id][1:]   
            self.p_history[agent_id][4] = agent.get_property_value(c.attitude_phi_rad)
            self.q_history[agent_id][:4] = self.q_history[agent_id][1:]
            self.q_history[agent_id][4] = agent.get_property_value(c.attitude_theta_rad)
            self.r_history[agent_id][:4] = self.r_history[agent_id][1:]
            self.r_history[agent_id][4] = agent.get_property_value(c.attitude_psi_rad)
            self.pdot_history[agent_id][:4] = self.pdot_history[agent_id][1:]
            self.pdot_history[agent_id][4] = agent.get_property_value(c.velocities_p_rad_sec)
            self.qdot_history[agent_id][:4] = self.qdot_history[agent_id][1:]
            self.qdot_history[agent_id][4] = agent.get_property_value(c.velocities_q_rad_sec)
            self.rdot_history[agent_id][:4] = self.rdot_history[agent_id][1:]
            self.rdot_history[agent_id][4] = agent.get_property_value(c.velocities_r_rad_sec)
            
    def get_task_history_variables(self, env, agent_id):
        return np.array([self.p_history[agent_id][-self.step_num:], 
                         self.q_history[agent_id][-self.step_num:],
                         self.r_history[agent_id][-self.step_num:],
                         self.pdot_history[agent_id][-self.step_num:],
                         self.qdot_history[agent_id][-self.step_num:],
                         self.rdot_history[agent_id][-self.step_num:]])

    def calculate_task_variables(self, env, agent_id):
        current_position_h_sl_m = self.state_prop_vals[17]
        current_velocities_vc_mps = self.state_prop_vals[6]
        current_attitude_heading_true_rad = self.state_prop_vals[16]

        #construct the heading task observation.
        target_altitude = self.mission_prop_vals[0]
        target_speed = self.mission_prop_vals[1]
        target_heading = self.mission_prop_vals[2]


        delta_altitude = target_altitude - current_position_h_sl_m

        delta_speed = target_speed - current_velocities_vc_mps

        delta_heading = target_heading - current_attitude_heading_true_rad
        return np.asarray([delta_altitude, delta_speed, delta_heading])

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])
    
    def get_obs(self, env, agent_id):
        agent = env.agents[agent_id]
        self.state_prop_vals = np.array(agent.get_property_values(self.state_props))
        self.mission_prop_vals = np.array(agent.get_property_values(self.mission_props))
        self.action_prop_vals = np.array(agent.get_property_values(self.action_props))

        task_variables = self.calculate_task_variables(env, agent_id)
        task_variable_observations = self.task_variables_to_observations(task_variables)        
        state_variable_observations = self.state_variables_to_observations(self.state_prop_vals)
        action_variable_observations = self.action_variables_to_observations(self.action_prop_vals)
    
        obs = np.concatenate([task_variable_observations, state_variable_observations, action_variable_observations])
        return obs
    
    def _convert_to_mach(self, velocities):
        return velocities / 340 #unit: mach
    
    def _convert_to_km(self, distances):
        return distances / 1000
    
    def action_variables_to_observations(self, action_variables):
        return action_variables
    
    def state_variables_to_observations(self, state_variables):
        attitude = state_variables[0:3].copy()
        uvw = state_variables[3:6].copy()
        vc = state_variables[6:7].copy()
        pqr = state_variables[7:10].copy()
        uvw_dot = state_variables[10:13].copy()
        pqr_dot = state_variables[13:16].copy()
        heading_true = state_variables[16:17].copy()
        altitude = state_variables[17:18].copy()
        attitude = self._convert_to_sincos(attitude).flatten()
        heading_true = self._convert_to_sincos(heading_true).flatten()
        uvw = self._convert_to_mach(uvw)
        altitude = self._convert_to_km(altitude)
        return np.concatenate([uvw, attitude, pqr, uvw_dot, pqr_dot, heading_true, vc, altitude])
    
    def task_variables_to_observations(self, task_variables):
#        delta_altitude, delta_speed, delta_heading
        delta_altitude = self._convert_to_km(task_variables[0:1])
                                                  
        delta_speed = self._convert_to_mach(task_variables[1:2])
        delta_heading = self._convert_to_sincos(task_variables[2:3]).flatten()

        task_variables = np.concatenate([delta_altitude, 
                                     delta_speed, 
                                     delta_heading]) 
        return task_variables
    
    def normalize_action(self, env, agent_id, action):
        return action
