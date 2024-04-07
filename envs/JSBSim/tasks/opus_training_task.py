import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import SafeAltitudeReward, OpusHeadingReward, OpusWaypointReward, OpusWaypointPotentialReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
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
            OpusHeadingReward(self.config),
            #OpusWaypointReward(self.config),
            #OpusWaypointPotentialReward(self.config),
            SafeAltitudeReward(self.config),
        ]

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
            c.atmosphere_headwind_mps,           # 21. headwind (unit: m/s)
            c.position_h_agl_m                  # 22. altitude above ground level (unit: m)
        ]

        # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        self.mission_var = [
            c.current_task_id,
            c.task_1_type_id,
            c.task_2_type_id,
            c.travel_1_target_position_h_sl_m,
            c.travel_1_target_attitude_psi_rad,
            c.travel_1_target_velocities_vc_mps,
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
        ]
            # c.wp_2_1_target_position_h_sl_m,
            # c.wp_2_1_target_position_lat_geod_rad,
            # c.wp_2_1_target_position_long_gc_rad,
            # c.wp_2_1_target_velocities_v_north_mps,
            # c.wp_2_1_target_velocities_v_east_mps,
            # c.wp_2_1_target_velocities_v_down_mps,
            # c.wp_2_1_target_time_s,
            # c.wp_2_2_target_position_h_sl_m,
            # c.wp_2_2_target_position_lat_geod_rad,
            # c.wp_2_2_target_position_long_gc_rad,
            # c.wp_2_2_target_velocities_v_north_mps,
            # c.wp_2_2_target_velocities_v_east_mps,
            # c.wp_2_2_target_velocities_v_down_mps,
            # c.wp_2_2_target_time_s,            
            # c.search_area_1_x1_grid,
            # c.search_area_1_y1_grid,
            # c.search_area_1_x2_grid,
            # c.search_area_1_y2_grid,
            # c.search_area_1_target_time_s,
            # c.search_area_2_x1_grid,
            # c.search_area_2_y1_grid,
            # c.search_area_2_x2_grid,
            # c.search_area_2_y2_grid,
            # c.search_area_2_target_time_s,
            # c.akan_destroy_1_target_id,
            # c.akan_destroy_1_target_time_s,
            # c.akan_destroy_2_target_id,
            # c.akan_destroy_2_target_time_s,
        #]

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
        # self.action_space = spaces.Box(
        #     low=np.asarray([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)
        #     ,high=np.asarray([1.0, 1.0, 1.0, 0.9], dtype=np.float32))

    def reset(self, env):
        super().reset(env)
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
        position_lla = np.asarray([state[1], state[2], state[0]])
        position_ned = LLA2NED(state[1], state[2], state[0], self.lat0, self.lon0, self.alt0)
        velocity_ned = np.asarray([state[3], state[4], state[5]])
        velocity_uvw = np.asarray([state[6], state[7], state[8]])
        acceleration_uvw = np.asarray([state[9], state[10], state[11]])
        attitude_w_x_y_z = self._convert_to_quaternion(state[12], state[13], state[14])
        attitude_rate_w_x_y_z = np.asarray([0, state[15], state[16], state[17]])
        heading = np.asarray([np.cos(state[18]), np.sin(state[18])])
        vc_u = np.asarray([state[19]])
        winds = np.asarray([state[20], state[21]])
        h_agl = np.asarray([state[22]])
        return np.concatenate([position_ned, velocity_ned, velocity_uvw, 
                               acceleration_uvw, attitude_w_x_y_z, 
                               attitude_rate_w_x_y_z, heading, vc_u, winds, h_agl])
    
    
    def normalize_angle_diff(self, angle_diff):
        """
        Normalize an angle difference to the range [-pi, pi].
        """
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
#     def calculate_waypoint_task_variables(self, current_time, env, agent_id):
#         # task type id
#         # 0: no mission (not used),
#         # 1: travel in heading at altitude and speed
#         # 2: travel to waypoint
#         # 3: search area
#         # 4: engage target
#         #todo: use agent_id
#         active_task = int(env.agents[agent_id].get_property_value(c.current_task_id))
#         if active_task == 1:
#             active_task_type = int(env.agents[agent_id].get_property_value(c.task_1_type_id))
#         else:
#             active_task_type = int(env.agents[agent_id].get_property_value(c.task_2_type_id))

#         if active_task_type != 2:
#             self.delta_1_north = np.array([0])
#             self.delta_1_east = np.array([0])
#             self.delta_1_down = np.array([0])
#             self.delta_1_v_north = np.array([0])
#             self.delta_1_v_east = np.array([0])
#             self.delta_1_v_down = np.array([0])
#             self.delta_1_time = np.array([0])
#             self.delta_2_north = np.array([0])
#             self.delta_2_east = np.array([0])
#             self.delta_2_down = np.array([0])
#             self.delta_2_v_north = np.array([0])
#             self.delta_2_v_east = np.array([0])
#             self.delta_2_v_down = np.array([0])
#             self.delta_2_time = np.array([0])
#         else:
#             self.delta_1_v_north = np.array([0])
#             self.delta_1_v_east = np.array([0])
#             self.delta_1_v_down = np.array([0])
#             self.delta_1_time = np.array([0])
#             self.delta_2_v_north = np.array([0])
#             self.delta_2_v_east = np.array([0])
#             self.delta_2_v_down = np.array([0])
#             self.delta_2_time = np.array([0])

#             ned_target = LLA2NED(self.mission_vars[12] * 180 / np.pi, self.mission_vars[13] * 180 / np.pi, self.mission_vars[11], self.lat0, self.lon0, self.alt0)
#             #construct the waypoint task observation.
#             self.delta_1_north = np.array([ned_target[0] - self.ned_frame_state[0]])
#             self.delta_1_east = np.array([ned_target[1] - self.ned_frame_state[1]])
#             self.delta_1_down = np.array([ned_target[2] - self.ned_frame_state[2]])
# #            self.delta_1_v_north = np.array([self.mission_vars[14] - self.ned_frame_state[3]])
# #            self.delta_1_v_east = np.array([self.mission_vars[15] - self.ned_frame_state[4]])
# #            self.delta_1_v_down = np.array([self.mission_vars[16] - self.ned_frame_state[5]])
# #            self.delta_1_time = np.array([self.mission_vars[17] - current_time])

#             ned_target = LLA2NED(self.mission_vars[19] * 180 / np.pi, self.mission_vars[20] * 180 / np.pi, self.mission_vars[18], self.lat0, self.lon0, self.alt0)
#             #construct the waypoint task observation.
#             self.delta_2_north = np.array([ned_target[0] - self.ned_frame_state[0]])
#             self.delta_2_east = np.array([ned_target[1] - self.ned_frame_state[1]])
#             self.delta_2_down = np.array([ned_target[2] - self.ned_frame_state[2]])
# #            self.delta_2_v_north = np.array([self.mission_vars[21] - self.ned_frame_state[3]])
# #            self.delta_2_v_east = np.array([self.mission_vars[22] - self.ned_frame_state[4]])
# #            self.delta_2_v_down = np.array([self.mission_vars[23] - self.ned_frame_state[5]])
# #            self.delta_2_time = np.array([self.mission_vars[24] - current_time])




    def calculate_heading_task_variables(self, env, agent_id):
        # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        #todo: use agent_id
        # active_task = int(env.agents[agent_id].get_property_value(c.current_task_id))

        # if active_task == 1:
        #     active_task_type = int(env.agents[agent_id].get_property_value(c.task_1_type_id))
        # else:
        #     active_task_type = int(env.agents[agent_id].get_property_value(c.task_2_type_id))

        # if active_task_type != 1:
        #     self.delta_altitude = np.array([0])
        #     self.delta_speed = np.array([0])
        #     self.delta_heading = np.array([0, 0])
        #     self.delta_time = np.array([0])
        # else:
        #     #ned_target = LLA2NED(self.state_props[1], self.state_props[2], self.mission_vars[3], self.lat0, self.lon0, self.alt0)
 
        #construct the heading task observation.
        target_altitude = np.asarray([self.mission_vars[3]])
        target_heading = np.asarray([self.mission_vars[4]])
        target_speed = np.asarray([self.mission_vars[5]])
        #convert to relative values.
        self.delta_altitude = target_altitude - self.state_props[0]
        delta_heading_rad = self.normalize_angle_diff(target_heading - self.state_props[14])
        self.delta_heading = np.asarray([np.cos(delta_heading_rad[0]), np.sin(delta_heading_rad[0])])
        self.delta_speed = target_speed - self.state_props[19]
        #what do we do about time?
        #target_time = self.mission_vars[6]
        #self.delta_time = np.asarray([target_time - current_time])
        return self.delta_altitude, self.delta_heading, self.delta_speed
    
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

        #mission_inputs:
        #self.mission_declarations = np.asarray([self.mission_vars[0], self.mission_vars[1], self.mission_vars[2]])
        #for now, train on one thing at a time.
        #we don't know what inputs are required so lets construct them one step at a time.
        #for now, we assume mission_vars[0] is always 0.
        #mission_vars[1] could be 1 or 2:
            # 1: travel in heading at altitude and speed
            # 2: travel to waypoint
        # current_time = agent.get_property_value(c.simulation_sim_time_sec)

        delta_altitude, delta_heading, delta_speed = self.calculate_heading_task_variables(env, agent_id)
        # self.calculate_waypoint_task_variables(current_time, env, agent_id)
        # waypoint_1_mission_vars = np.concatenate([self.delta_1_north, 
        #                                         self.delta_1_east, 
        #                                         self.delta_1_down, 
        #                                         self.delta_1_v_north, 
        #                                         self.delta_1_v_east, 
        #                                         self.delta_1_v_down, 
        #                                         self.delta_1_time])
        # waypoint_2_mission_vars = np.concatenate([
        #                                         self.delta_2_north, 
        #                                         self.delta_2_east, 
        #                                         self.delta_2_down, 
        #                                         self.delta_2_v_north, 
        #                                         self.delta_2_v_east, 
        #                                         self.delta_2_v_down, 
        #                                         self.delta_2_time])
        
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

        # # norm_obs[25] = self.ned_frame_state[3] / 340  # 6. v_north (unit: mach)
        # # norm_obs[26] = self.ned_frame_state[4] / 340 # 7. v_east (unit: mach)
        # # norm_obs[27] = self.ned_frame_state[5] / 340 # 8. v_down (unit: mach)
        # # norm_obs[28] = self.ned_frame_state[6] / 340 # 9. u (unit: mach)
        # # norm_obs[29] = self.ned_frame_state[7] / 340 # 10. v (unit: mach)
        # # norm_obs[30] = self.ned_frame_state[8] / 340 # 11. w (unit: mach)
        # # norm_obs[31] = self.ned_frame_state[9] / 340 # 12. udot (unit: mach)
        # # norm_obs[32] = self.ned_frame_state[10] / 340    # 13. vdot (unit: mach)
        # # norm_obs[33] = self.ned_frame_state[11] / 340    # 14. wdot (unit: mach)
        # # norm_obs[34] = self.ned_frame_state[12]          # 15. attitude_w (quat.w)
        # # norm_obs[35] = self.ned_frame_state[13]          # 16. attitude_x (quat.x)
        # # norm_obs[36] = self.ned_frame_state[14]          # 17. attitude_y (quat.y)
        # # norm_obs[37] = self.ned_frame_state[15]          # 18. attitude_z (quat.z)
        # # norm_obs[38] = self.ned_frame_state[16]          # 19. attitude_rate_w (quat.w)
        # # norm_obs[39] = self.ned_frame_state[17]          # 20. attitude_rate_x (quat.x)
        # # norm_obs[40] = self.ned_frame_state[18]          # 21. attitude_rate_y (quat.y)
        # # norm_obs[41] = self.ned_frame_state[19]          # 22. attitude_rate_z (quat.z)
        # norm_obs[42] = self.ned_frame_state[20]          # 23. heading (cos)
        # norm_obs[43] = self.ned_frame_state[21]          # 24. heading (sin)
        # norm_obs[44] = self.ned_frame_state[22] / 340    # 25. vc (unit: mach)
        # #norm_obs[45] = self.ned_frame_state[23] / 5000    # 26. crosswind (unit: 5 km/s)
        # #norm_obs[46] = self.ned_frame_state[24] / 5000    # 27. headwind (unit: 5 km/s)
        # #norm_obs[47] = self.ned_frame_state[25] / 100    # 28. latitude (unit: 100 deg)
        # #norm_obs[48] = self.ned_frame_state[26] / 100    # 29. longitude (unit: 100 deg)
        # #norm_obs[49] = self.ned_frame_state[27] / 5000    # 30. altitude (unit: 5 km)
        # norm_obs[47] = (self.ned_frame_state[25] - 500) / 5000    # 31. altitude above safe ground level (unit: 5 km)
        # norm_obs[48] = self.action_props[0]              # 28. aileron cmd norm
        # norm_obs[49] = self.action_props[1]              # 29. elevator cmd norm
        # norm_obs[50] = self.action_props[2]              # 30. rudder cmd norm
        # norm_obs[51] = self.action_props[3]              # 31. throttle cmd norm
        norm_obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        return action
