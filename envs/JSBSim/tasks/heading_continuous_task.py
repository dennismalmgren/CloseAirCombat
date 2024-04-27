import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading


class HeadingTaskContinuous(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
        ]
        self.truncation_condition = Timeout(self.config)

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        
        self.state_var_jsbsim = [
            c.position_h_sl_m,                  # altitude  (unit: m)
            c.position_lat_geod_rad,            # latitude geodetic (unit: rad)
            c.position_long_gc_rad,             # longitude geocentric (same as geodetic) (unit: rad)
            c.velocities_v_north_mps,           # v_north    (unit: m/s)
            c.velocities_v_east_mps,            # v_east     (unit: m/s)
            c.velocities_v_down_mps,            # v_down     (unit: m/s)
            c.velocities_u_mps,                 # v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # v_body_z   (unit: m/s)
            c.accelerations_udot_m_sec2,        # body_x acceleration (unit: m/s),
            c.accelerations_vdot_m_sec2,        # body_y acceleration (unit: m/s),
            c.accelerations_wdot_m_sec2,        # body_z acceleration (unit: m/s),
            c.attitude_phi_rad,                 # roll      (unit: rad)
            c.attitude_theta_rad,               # pitch     (unit: rad)
            c.attitude_psi_rad,                 # yaw     (unit: rad)
            c.velocities_p_rad_sec,             # roll rate (unit: rad)
            c.velocities_q_rad_sec,             # pitch rate (unit: rad)
            c.velocities_r_rad_sec,             # yaw rate (unit: rad)
            c.attitude_heading_true_rad,        # heading (unit: rad)
            c.velocities_vc_mps,                # vc        (unit: m/s)
        ]

        # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        self.mission_var_jsbsim = [
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
            c.wp_1_target_position_h_sl_m,
            c.wp_1_target_position_lat_geod_rad,
            c.wp_1_target_position_long_gc_rad,
            c.wp_1_target_velocities_v_north_mps,
            c.wp_1_target_velocities_v_east_mps,
            c.wp_1_target_velocities_v_down_mps,
            c.wp_1_target_time_s,
            c.wp_2_target_position_h_sl_m,
            c.wp_2_target_position_lat_geod_rad,
            c.wp_2_target_position_long_gc_rad,
            c.wp_2_target_velocities_v_north_mps,
            c.wp_2_target_velocities_v_east_mps,
            c.wp_2_target_velocities_v_down_mps,
            c.wp_2_target_time_s,
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

        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
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

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(4,))
        # self.action_space = spaces.Box(
        #     low=np.asarray([-1.0, -1.0, -1.0, 0.4], dtype=np.float32)
        #     ,high=np.asarray([1.0, 1.0, 1.0, 0.9], dtype=np.float32))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        state_obs = np.array(env.agents[agent_id].get_property_values(self.state_var_jsbsim))
        mission_obs = np.array(env.agents[agent_id].get_property_values(self.mission_var_jsbsim))

        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        return action
