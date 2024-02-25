import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward, SafeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..curricula import OpusCurriculum
from ..utils.utils import LLA2NED, NED2LLA

class OpusTrainingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        self.lat0, self.lon0, self.alt0 = config.battle_field_origin
        self.n0, self.e0, self.u0 = 0, 0, 0

        self.reward_functions = [
            #HeadingReward(self.config),
            SafeAltitudeReward(self.config),
        ]
        self.curriculum = OpusCurriculum(self.config)

        self.termination_conditions = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
        ]


        self.truncation_condition = Timeout(self.config)

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.position_h_sl_m,                  # 0. altitude  (unit: m)
            c.position_lat_geod_deg,            # 1. latitude geodetic (unit: deg)
            c.position_long_gc_deg,             # 2. longitude geocentric (same as geodetic) (unit: deg)
            c.velocities_v_north_mps,           # 3. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 4. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 5. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.accelerations_udot_m_sec2,        # 9. body_x acceleration (unit: m/s),
            c.accelerations_vdot_m_sec2,        # 10. body_y acceleration (unit: m/s),
            c.accelerations_wdot_m_sec2,        # 11. body_z acceleration (unit: m/s),
            c.attitude_phi_rad,                 # 12. roll      (unit: rad)
            c.attitude_theta_rad,               # 13. pitch     (unit: rad)
            c.attitude_psi_rad,                 # 14. yaw     (unit: rad)
            c.velocities_p_rad_sec,             # 15. roll rate (unit: rad)
            c.velocities_q_rad_sec,             # 16. pitch rate (unit: rad)
            c.velocities_r_rad_sec,             # 17. yaw rate (unit: rad)
            c.attitude_heading_true_rad,        # 18. heading (unit: rad)
            c.velocities_vc_mps,                # 19. vc        (unit: m/s)
            c.atmosphere_crosswind_mps,         # 20. crosswind (unit: m/s)
            c.atmosphere_headwind_mps           # 21. headwind (unit: m/s)
        ]

        # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        self.mission_var = [
            c.task_1_type_id,
            c.task_2_type_id,
            c.travel_1_target_position_h_sl_m,
            c.travel_1_target_attitude_psi_rad,
            c.travel_1_target_velocities_u_mps,
            c.travel_1_target_time_s,
            c.travel_2_target_position_h_sl_m,
            c.travel_2_target_attitude_psi_rad,
            c.travel_2_target_velocities_u_mps,
            c.travel_2_target_time_s,
            c.wp_1_1_target_position_h_sl_m,
            c.wp_1_1_target_position_lat_geod_rad,
            c.wp_1_1_target_position_long_gc_rad,
            c.wp_1_1_target_velocities_v_north_mps,
            c.wp_1_1_target_velocities_v_east_mps,
            c.wp_1_1_target_velocities_v_down_mps,
            c.wp_1_1_target_time_s,
            c.wp_1_2_target_position_h_sl_m,
            c.wp_1_2_target_position_lat_geod_rad,
            c.wp_1_2_target_position_long_gc_rad,
            c.wp_1_2_target_velocities_v_north_mps,
            c.wp_1_2_target_velocities_v_east_mps,
            c.wp_1_2_target_velocities_v_down_mps,
            c.wp_1_2_target_time_s,            
            c.wp_2_1_target_position_h_sl_m,
            c.wp_2_1_target_position_lat_geod_rad,
            c.wp_2_1_target_position_long_gc_rad,
            c.wp_2_1_target_velocities_v_north_mps,
            c.wp_2_1_target_velocities_v_east_mps,
            c.wp_2_1_target_velocities_v_down_mps,
            c.wp_2_1_target_time_s,
            c.wp_2_2_target_position_h_sl_m,
            c.wp_2_2_target_position_lat_geod_rad,
            c.wp_2_2_target_position_long_gc_rad,
            c.wp_2_2_target_velocities_v_north_mps,
            c.wp_2_2_target_velocities_v_east_mps,
            c.wp_2_2_target_velocities_v_down_mps,
            c.wp_2_2_target_time_s,            
            c.search_area_1_x1_grid,
            c.search_area_1_y1_grid,
            c.search_area_1_x2_grid,
            c.search_area_1_y2_grid,
            c.search_area_1_target_time_s,
            c.search_area_2_x1_grid,
            c.search_area_2_y1_grid,
            c.search_area_2_x2_grid,
            c.search_area_2_y2_grid,
            c.search_area_2_target_time_s,
            c.akan_destroy_1_target_id,
            c.akan_destroy_1_target_time_s,
            c.akan_destroy_2_target_id,
            c.akan_destroy_2_target_time_s,
        ]

        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(35,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(4,))
        # self.action_space = spaces.Box(
        #     low=np.asarray([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)
        #     ,high=np.asarray([1.0, 1.0, 1.0, 0.9], dtype=np.float32))

    def reset(self, env):
        super().reset(env)
        self.curriculum.reset(self, env)
#        agent = env.agents.values()[0]
#        #hack. move it from termination condition..
#        self.termination_conditions[0].reset(env, agent)
        
    
    def _convert_to_quaternion(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return np.asarray([w, x, y, z])
    
    def convert_state_to_ned_frame(self, state):
        position_ned = LLA2NED(state[1], state[2], state[0], self.lat0, self.lon0, self.alt0)
        velocity_ned = np.asarray([state[3], state[4], state[5]])
        velocity_uvw = np.asarray([state[6], state[7], state[8]])
        acceleration_uvw = np.asarray([state[9], state[10], state[11]])
        attitude_w_x_y_z = self._convert_to_quaternion(state[12], state[13], state[14])
        attitude_rate_w_x_y_z = np.asarray([0, state[15], state[16], state[17]])
        heading = np.asarray([np.cos(state[18]), np.sin(state[18])])
        vc_u = np.asarray([state[19]])
        winds = np.asarray([state[20], state[21]])
        return np.concatenate([position_ned, velocity_ned, velocity_uvw, 
                               acceleration_uvw, attitude_w_x_y_z, 
                               attitude_rate_w_x_y_z, heading, vc_u, winds])
    
    
    def normalize_angle_diff(self, angle_diff):
        """
        Normalize an angle difference to the range [-pi, pi].
        """
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.
        observation space:
        0. task 1 type id
        1. task 2 type id
        2. delta altitude ()
        ned frame state
        0. north
        1. east
        2. down
        3. v_north
        4. v_east
        5. v_down
        6. u
        7. v
        8. w
        9. udot
        10. vdot
        11. wdot
        12. attitude_w
        13. attitude_x
        14. attitude_y
        15. attitude_z
        16. attitude_rate_w
        17. attitude_rate_x
        18. attitude_rate_y
        19. attitude_rate_z
        20. heading (cos)
        21. heading (sin)
        22. vc
        23. crosswind
        24. headwind
        25. aileron cmd norm
        26. elevator cmd norm
        27. rudder cmd norm
        28. throttle cmd norm
        """
        state_props = np.array(env.agents[agent_id].get_property_values(self.state_var))
        
        ned_frame_state = self.convert_state_to_ned_frame(state_props)
        action_props = np.array(env.agents[agent_id].get_property_values(self.action_var))
        mission_vars = np.array(env.agents[agent_id].get_property_values(self.mission_var))
        #mission_inputs:
        tasks = np.asarray([mission_vars[0]])
        #convert target altitude to NED.
        ned_target = LLA2NED(state_props[1], state_props[2], mission_vars[2], self.lat0, self.lon0, self.alt0)

        # Constructing the heading task observation
        target_altitude = np.asarray([ned_target[2]])
        target_heading = np.asarray([mission_vars[3]])
        target_speed = np.asarray([mission_vars[4]])
        #convert to relative values.
        delta_altitude = target_altitude - ned_frame_state[2]
        delta_heading = self.normalize_angle_diff(target_heading - state_props[18])
        delta_heading = np.asarray([np.cos(delta_heading), np.sin(delta_heading)])
        delta_speed = target_speed - ned_frame_state[6]
        
        #lets build observation.
        norm_obs = np.zeros(35)

        norm_obs[0] = tasks[0]                  # task 1 type id
        #heading task representation
        norm_obs[2] = delta_altitude / 5000     # 2. delta altitude (unit: 5km)
        norm_obs[3] = delta_heading[0]          # 3. delta heading (cos)
        norm_obs[4] = delta_heading[1]          # 4. delta heading (sin)
        norm_obs[5] = delta_speed / 340         # 5. delta speed (unit: mach)
        #local state representation, for efficient aircraft handling
        norm_obs[6] = ned_frame_state[0] / 5000 # 3. north (unit: 5km)
        norm_obs[7] = ned_frame_state[1] / 5000 # 4. east (unit: 5km)
        norm_obs[8] = ned_frame_state[2] / 5000 # 5. down (unit: 5km)
        norm_obs[9] = ned_frame_state[3] / 340  # 6. v_north (unit: mach)
        norm_obs[10] = ned_frame_state[4] / 340 # 7. v_east (unit: mach)
        norm_obs[11] = ned_frame_state[5] / 340 # 8. v_down (unit: mach)
        norm_obs[12] = ned_frame_state[6] / 340 # 9. u (unit: mach)
        norm_obs[13] = ned_frame_state[7] / 340 # 10. v (unit: mach)
        norm_obs[14] = ned_frame_state[8] / 340 # 11. w (unit: mach)
        norm_obs[15] = ned_frame_state[9] / 340 # 12. udot (unit: mach)
        norm_obs[16] = ned_frame_state[10] / 340    # 13. vdot (unit: mach)
        norm_obs[17] = ned_frame_state[11] / 340    # 14. wdot (unit: mach)
        norm_obs[18] = ned_frame_state[12]          # 15. attitude_w (quat.w)
        norm_obs[19] = ned_frame_state[13]          # 16. attitude_x (quat.x)
        norm_obs[20] = ned_frame_state[14]          # 17. attitude_y (quat.y)
        norm_obs[21] = ned_frame_state[15]          # 18. attitude_z (quat.z)
        norm_obs[22] = ned_frame_state[16]          # 19. attitude_rate_w (quat.w)
        norm_obs[23] = ned_frame_state[17]          # 20. attitude_rate_x (quat.x)
        norm_obs[24] = ned_frame_state[18]          # 21. attitude_rate_y (quat.y)
        norm_obs[25] = ned_frame_state[19]          # 22. attitude_rate_z (quat.z)
        norm_obs[26] = ned_frame_state[20]          # 23. heading (cos)
        norm_obs[27] = ned_frame_state[21]          # 24. heading (sin)
        norm_obs[28] = ned_frame_state[22] / 340    # 25. vc (unit: mach)
        norm_obs[29] = ned_frame_state[23] / 5000    # 26. crosswind (unit: 5 km/s)
        norm_obs[30] = ned_frame_state[24] / 5000    # 27. headwind (unit: 5 km/s)
        norm_obs[31] = action_props[0]              # 28. aileron cmd norm
        norm_obs[32] = action_props[1]              # 29. elevator cmd norm
        norm_obs[33] = action_props[2]              # 30. rudder cmd norm
        norm_obs[34] = action_props[3]              # 31. throttle cmd norm
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        return action
