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
            c.position_h_sl_m,                  # 0. altitude  (unit: m)
            c.velocities_u_mps,                 # 1. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 2. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 3. v_body_z   (unit: m/s)
            c.attitude_phi_rad,                 # 4. roll      (unit: rad)
            c.attitude_theta_rad,               # 5. pitch     (unit: rad)
            c.attitude_psi_rad,                 # 6. yaw     (unit: rad)
            c.velocities_vc_mps,                # 7. vc        (unit: m/s)
            c.attitude_heading_true_rad,         # 8. heading   (unit: rad)
            c.accelerations_udot_m_sec2,
            c.accelerations_vdot_m_sec2,
            c.accelerations_wdot_m_sec2,
            c.accelerations_pdot_rad_sec2,
            c.accelerations_qdot_rad_sec2,
            c.accelerations_rdot_rad_sec2,
            c.velocities_h_dot_mps
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
        state_variable_count = 28
        self.observation_space = spaces.Box(low=-10, high=10., shape=(task_variable_count + state_variable_count,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=np.asarray([-1.0, -1.0, -1.0, 0.0]),
                                       high=np.asarray([1.0, 1.0, 1.0, 0.9]), dtype=np.float32, shape=(4,))

    def reset(self, env):
        super().reset(env)
        self.step_num = 1
        self.reset_smoothness(env)
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
        self.step_smoothness(env)
        self.step_num += 1

    def reset_task_history(self, env):
        self.w_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.p_history = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.step_task_history(env)

    def step_task_history(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.w_history[agent_id][:4] = self.w_history[agent_id][1:]         
            self.p_history[agent_id][:4] = self.p_history[agent_id][1:]   
            self.w_history[agent_id][4] = agent.get_property_value(c.velocities_w_mps) / 340.0
            self.p_history[agent_id][4] = agent.get_property_value(c.attitude_phi_rad)

    def step_smoothness(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.smoothness_u[agent_id][:4] = self.smoothness_u[agent_id][1:]
            self.smoothness_v[agent_id][:4] = self.smoothness_v[agent_id][1:]
            self.smoothness_w[agent_id][:4] = self.smoothness_w[agent_id][1:]
            self.smoothness_p[agent_id][:4] = self.smoothness_p[agent_id][1:]
            self.smoothness_q[agent_id][:4] = self.smoothness_q[agent_id][1:]
            self.smoothness_r[agent_id][:4] = self.smoothness_r[agent_id][1:]
            self.smoothness_u[agent_id][4] = agent.get_property_value(c.accelerations_udot_ft_sec2) * 0.3048
            self.smoothness_v[agent_id][4] = agent.get_property_value(c.accelerations_vdot_ft_sec2) * 0.3048
            self.smoothness_w[agent_id][4] = agent.get_property_value(c.accelerations_wdot_ft_sec2) * 0.3048
            self.smoothness_p[agent_id][4] = agent.get_property_value(c.accelerations_pdot_rad_sec2)
            self.smoothness_q[agent_id][4] = agent.get_property_value(c.accelerations_qdot_rad_sec2)
            self.smoothness_r[agent_id][4] = agent.get_property_value(c.accelerations_rdot_rad_sec2)

    def reset_smoothness(self, env):
        self.smoothness_u = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.smoothness_v = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.smoothness_w = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.smoothness_p = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.smoothness_q = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.smoothness_r = {agent_id: np.zeros(5) for agent_id in env.agents}
        self.step_smoothness(env)

    def get_task_history_variables(self, env, agent_id):
        return np.array([self.w_history[agent_id][-self.step_num:], 
                         self.p_history[agent_id][-self.step_num:]])
    
    def get_smoothness_variables(self, env, agent_id):
        return np.array([self.smoothness_u[agent_id][-self.step_num:], 
                         self.smoothness_v[agent_id][-self.step_num:], 
                         self.smoothness_w[agent_id][-self.step_num:], 
                         self.smoothness_p[agent_id][-self.step_num:], 
                         self.smoothness_q[agent_id][-self.step_num:], 
                         self.smoothness_r[agent_id][-self.step_num:]])
        
    def calculate_smoothness(self, env, agent_id):
        if self.step_num < 5:
            return np.zeros(6)
        smoothness_u = self.smoothness_u[agent_id]
        smoothness_v = self.smoothness_v[agent_id]
        smoothness_w = self.smoothness_w[agent_id]
        smoothness_p = self.smoothness_p[agent_id]
        smoothness_q = self.smoothness_q[agent_id]
        smoothness_r = self.smoothness_r[agent_id]
        #this is the second derivatives. we derive 4 third derivatives
        smoothness_u = np.diff(smoothness_u, n=4)
        smoothness_v = np.diff(smoothness_v, n=4)
        smoothness_w = np.diff(smoothness_w, n=4)
        smoothness_p = np.diff(smoothness_p, n=4)
        smoothness_q = np.diff(smoothness_q, n=4)
        smoothness_r = np.diff(smoothness_r, n=4)
        return np.array([smoothness_u, smoothness_v, smoothness_w, smoothness_p, smoothness_q, smoothness_r])
    
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

        task_variables = self.calculate_task_variables(env, agent_id)
        task_variables = self.transform_task_variables(task_variables)        
        altitude = self.state_prop_vals[0:1].copy()
        altitude /= 5000 #unit: 5km
        uvw = self.state_prop_vals[1:4].copy()
        uvw = self.transform_uvw(uvw)
        attitude = self.state_prop_vals[4:7].copy()
        attitude = self._convert_to_sincos(attitude).flatten()
        speed_vc = self.state_prop_vals[7:8].copy()
        speed_vc /= 340 #unit: mach
        attitude_heading = self.state_prop_vals[8]
        attitude_heading = self._convert_to_sincos(attitude_heading)
        uvw_acc = self.state_prop_vals[9:12].copy()
        uvw_acc = self.transform_uvw(uvw_acc)
        #pqr_in = self.state_prop_vals[12:15].copy()
        #pqr_acc = np.zeros(4,)
        #pqr_acc[1:] = pqr_in
        #pqr_quat = 0.5 * self.quaternion_multiply(attitude, pqr_acc)
        pqr_in = self.state_prop_vals[12:15].copy()
        pqr_in = self._convert_to_sincos(pqr_in).flatten()
        altitude_v = self.state_prop_vals[15:16].copy()
        altitude_v = self.transform_uvw(altitude_v)
        action_variables = self.get_action_variables(env, agent_id) 
        phi_rad = self.state_prop_vals[4:5].copy()
        obs = np.concatenate([phi_rad, uvw, attitude, pqr_in, uvw_acc, attitude_heading, speed_vc, altitude, altitude_v, action_variables, task_variables])

        norm_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return norm_obs
    
    def get_action_variables(self, env, agent_id):
        agent = env.agents[agent_id]
        action_variables = np.array(agent.get_property_values(self.action_props))
        return action_variables
    
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
