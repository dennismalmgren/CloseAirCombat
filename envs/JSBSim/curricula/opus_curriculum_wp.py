import math
from ..core.catalog import Catalog as c
from .curiculum_base import BaseCurriculum
import numpy as np
from ..utils.utils import LLA2NED, NED2LLA
from ..core.trajectory_planning import FlightModes, FlightProperties, Waypoint, FlightDirective

class OpusCurriculumWaypoints(BaseCurriculum):
    """
    Updates the current opus training task.
    """
    def __init__(self, config):
        super().__init__(config)
        self.waypoints = None
        # self.min_waypoint_distance = 1000 #m
        # self.min_waypoint_altitude = 1000 #m 
        # self.max_waypoint_altitude = 10000 #m
        # self.min_north = 0 #m
        # self.max_north = 200_000 #m
        # self.min_east = 0 #m
        # self.max_east = 400_000 #m
        # self.min_speed_x = 100  #m/s
        # self.max_speed_x = 450  #m/s
        # self.min_speed_y = 100  #m/s
        # self.max_speed_y = 450  #m/s
        # self.min_speed_z = 0  #m/s
        # self.max_speed_z = 50  #m/s


    def get_init_state(self, agent_id):
        #hack. we know it's only one agent for now..
        return self.agent_init_states[agent_id]
    
    def create_init_states(self, env):
        agent_init_states = dict()
        for agent_id in env.agents:
            
            init_heading_deg = env.np_random.uniform(0., 180.)
            init_altitude_m = env.np_random.uniform(2500., 9000.)
            init_velocities_u_mps = env.np_random.uniform(120., 365.)

            agent_init_states[agent_id] = {
                'ic_psi_true_deg': init_heading_deg,
                'ic_h_sl_ft': init_altitude_m / 0.3048,
                'ic_u_fps': init_velocities_u_mps / 0.3048,
            }
        return agent_init_states
    
    # def create_random_waypoint(self, agent, env):
    #     agent.set_property_value(c.current_task_id, 1) #1 or 2. we always go with 0 for now..
    #     agent.set_property_value(c.task_1_type_id, 2) #1 for heading, 2 for waypoint.
    #     agent.set_property_value(c.task_2_type_id, 0) #0 for no mission.

    #     current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.

    #     current_altitude = agent.get_property_value(c.position_h_sl_m)
    #     current_lat_deg = agent.get_property_value(c.position_lat_geod_deg)
    #     current_lon_deg = agent.get_property_value(c.position_long_gc_deg)

    #     current_location_ned = LLA2NED(current_lat_deg, current_lon_deg, current_altitude, env.task.lat0, env.task.lon0, env.task.alt0)
    #     current_location_neu = np.array([current_location_ned[0], current_location_ned[1], -current_location_ned[2]])

    #     waypoint_north = env.np_random.uniform(self.min_north, self.max_north)
    #     waypoint_east = env.np_random.uniform(self.min_east, self.max_east)
    #     waypoint_altitude = env.np_random.uniform(self.min_waypoint_altitude, self.max_waypoint_altitude)
        
    #     flightmode_choice = env.np_random.integers(0, 3)
    #     if flightmode_choice == 0:
    #         flightmode = FlightModes.LOITER
    #     elif flightmode_choice == 1:
    #         flightmode = FlightModes.CRUISE
    #     else:
    #         flightmode = FlightModes.ENGAGE
    
    #     target_location_neu = np.array([waypoint_north, waypoint_east, waypoint_altitude])
    #     target_velocity = np.array([env.np_random.uniform(self.min_speed_x, self.max_speed_x), env.np_random.uniform(self.min_speed_y, self.max_speed_y), env.np_random.uniform(self.min_speed_z, self.max_speed_z)])

    #     #target_location_neu = np.array([100000.0,      0.0,   4000.0])
    #     #target_velocity = np.array([     0.0,    200.0,    100.0])
    #     #current_location_neu = np.array([     0.0,      0.0,   10000.0])

    #     wp = Waypoint(target_location_neu, target_velocity)
    #     fd = FlightDirective(current_location_neu, flightmode, wp)
    #     p_and_t = fd.path_and_time()
    #     wp0 = p_and_t[0]
    #     wp0_ned = wp0[:3] #actually neu
    #     wp0_ned[2] = -wp0_ned[2]
    #     wp0_ned[0], wp0_ned[1], wp0_ned[2] = NED2LLA(wp0_ned[0], wp0_ned[1], wp0_ned[2], env.task.lat0, env.task.lon0, env.task.alt0)
    #     agent.set_property_value(c.wp_1_1_target_position_h_sl_m, wp0_ned[2])
    #     agent.set_property_value(c.wp_1_1_target_position_lat_geod_rad, wp0_ned[0] * np.pi / 180)
    #     agent.set_property_value(c.wp_1_1_target_position_long_gc_rad, wp0_ned[1] * np.pi / 180)
    #     agent.set_property_value(c.wp_1_1_target_velocities_v_north_mps, wp0[3])
    #     agent.set_property_value(c.wp_1_1_target_velocities_v_east_mps, wp0[4])
    #     agent.set_property_value(c.wp_1_1_target_velocities_v_down_mps, -wp0[5])            
    #     agent.set_property_value(c.wp_1_1_target_time_s, current_time + wp0[6])
    #     wp1 = p_and_t[1]
    #     wp1_ned = wp1[:3] #actually neu
    #     wp1_ned[2] = -wp1_ned[2]
    #     wp1_ned[0], wp1_ned[1], wp1_ned[2] = NED2LLA(wp1_ned[0], wp1_ned[1], wp1_ned[2], env.task.lat0, env.task.lon0, env.task.alt0)
    #     agent.set_property_value(c.wp_1_2_target_position_h_sl_m, wp1_ned[2])
    #     agent.set_property_value(c.wp_1_2_target_position_lat_geod_rad, wp1_ned[0] * np.pi / 180)
    #     agent.set_property_value(c.wp_1_2_target_position_long_gc_rad, wp1_ned[1] * np.pi / 180)
    #     agent.set_property_value(c.wp_1_2_target_velocities_v_north_mps, wp1[3])
    #     agent.set_property_value(c.wp_1_2_target_velocities_v_east_mps, wp1[4])
    #     agent.set_property_value(c.wp_1_2_target_velocities_v_down_mps, -wp1[5])
    #     agent.set_property_value(c.wp_1_2_target_time_s, current_time + wp1[6])
    
    def create_waypoints(self, env):
        for agent in env.agents.values():
            agent_alt = agent.get_property_value(c.position_h_sl_m)                  # 0. altitude  (unit: m)
            agent_lat = agent.get_property_value(c.position_lat_geod_deg)            # 1. latitude geodetic (unit: deg)
            agent_lon = agent.get_property_value(c.position_long_gc_deg)           
            current_north, current_east, current_down = LLA2NED(agent_lat, agent_lon, agent_alt, env.task.lat0, env.task.lon0, env.task.alt0)
            wp1 = (current_north, current_east, current_down)
            wp2 = (current_north, current_east + 2000, current_down + 1000)
            wp3 = (current_north + 2000, current_east, current_down - 1000)
            wp4 = (current_north + 2000, current_east+2000, current_down)
            self.waypoints = [wp1, wp2, wp3, wp4]
            self.mach_numbers = [0.4, 0.6, 0.8]

    def select_next_waypoint(self, env, current_waypoint_ind):
        available_waypoint_indices = set(np.arange(0, 4))
        available_waypoint_indices.remove(current_waypoint_ind)
        
        available_waypoint_indices = list(available_waypoint_indices)
        available_waypoint_indices = env.np_random.permutation(available_waypoint_indices)

        next_waypoint_ind_ind = env.np_random.integers(0, 3)
        next_waypoint_ind = available_waypoint_indices[next_waypoint_ind_ind]
        return next_waypoint_ind

    # def set_waypoint_task(self, env, agent, waypoint_ind, task_ind):
    #     if task_ind == 1:

    def update_waypoint_1(self, waypoint, mach_limit, env, agent):
        lla1 = NED2LLA(waypoint[0], waypoint[1], waypoint[2], 
                          env.task.lat0, env.task.lon0, env.task.alt0)
            
        agent.set_property_value(c.wp_1_1_target_position_h_sl_m, lla1[2])
        agent.set_property_value(c.wp_1_1_target_position_lat_geod_rad, lla1[0] * np.pi / 180.0)
        agent.set_property_value(c.wp_1_1_target_position_long_gc_rad, lla1[1] * np.pi / 180.0)
        agent.set_property_value(c.travel_1_target_velocities_u_mps, mach_limit * 340)

    def update_waypoint_2(self, waypoint, mach_limit, env, agent):
        lla2 = NED2LLA(waypoint[2], waypoint[1], waypoint[2], 
                          env.task.lat0, env.task.lon0, env.task.alt0)
        agent.set_property_value(c.wp_1_2_target_position_h_sl_m, lla2[2])
        agent.set_property_value(c.wp_1_2_target_position_lat_geod_rad, lla2[0] * np.pi / 180.0)
        agent.set_property_value(c.wp_1_2_target_position_long_gc_rad, lla2[1] * np.pi / 180.0)
        agent.set_property_value(c.travel_2_target_velocities_u_mps, mach_limit * 340)

    def reset(self, env):
        self.create_waypoints(env)
        for agent in env.agents.values():
            agent.set_property_value(c.current_task_id, 1)
            agent.set_property_value(c.task_1_type_id, 2) #1 for heading, 2 for waypoint.
            agent.set_property_value(c.task_2_type_id, 2) #0 for no mission.

            current_waypoint_ind = env.np_random.integers(1, 4)            
            current_waypoint = self.waypoints[current_waypoint_ind]
            current_mach_limit_ind = env.np_random.integers(0, 3)
            current_mach_limit = self.mach_numbers[current_mach_limit_ind]

            next_waypoint_ind = self.select_next_waypoint(env, current_waypoint_ind)
            next_mach_limit_ind = env.np_random.integers(0, 3)
            next_waypoint = self.waypoints[next_waypoint_ind]
            next_mach_limit = self.mach_numbers[next_mach_limit_ind]
            self.target_waypoint_inds = [current_waypoint_ind, next_waypoint_ind]
            self.target_waypoints = [current_waypoint, next_waypoint]
            self.target_mach_limits = [current_mach_limit, next_mach_limit]

            self.update_waypoint_1(self.target_waypoints[0], self.target_mach_limits[0], env, agent)
            self.update_waypoint_2(self.target_waypoints[1], self.target_mach_limits[1], env, agent)
         
            
    def step(self, env, agent_id, info= {}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        agent = env.agents[agent_id]
        agent_alt = agent.get_property_value(c.position_h_sl_m)                  # 0. altitude  (unit: m)
        agent_lat = agent.get_property_value(c.position_lat_geod_deg)            # 1. latitude geodetic (unit: deg)
        agent_lon = agent.get_property_value(c.position_long_gc_deg)           
        current_north, current_east, current_down = LLA2NED(agent_lat, agent_lon, agent_alt, env.task.lat0, env.task.lon0, env.task.alt0)
        active_task = int(agent.get_property_value(c.current_task_id))
        active_waypoint_index = active_task - 1
        if active_task == 1:
            active_task_type = int(agent.get_property_value(c.task_1_type_id))
        else:
            active_task_type = int(agent.get_property_value(c.task_2_type_id))

        if active_task_type != 2:
            return
        else:
            dist = np.linalg.norm(np.array([current_north - self.waypoints[active_waypoint_index][0], 
                                current_east - self.waypoints[active_waypoint_index][1], 
                                current_down - self.waypoints[active_waypoint_index][2]]))
            if dist < 100:
                current_task_id = int(agent.get_property_value(c.current_task_id))
                current_waypoint_ind = self.target_waypoint_inds[int(current_task_id) - 1]
                next_waypoint_ind = self.select_next_waypoint(env, current_waypoint_ind)
                next_mach_ind = env.np_random.integers(0, 3)
                if current_task_id == 1:
                    self.update_waypoint_1(self.waypoints[next_waypoint_ind], self.mach_numbers[next_mach_ind], env, agent)

                    agent.set_property_value(c.current_task_id, 3 - current_task_id)
                else:
                    self.update_waypoint_2(self.waypoints[next_waypoint_ind], self.mach_numbers[next_mach_ind], env, agent)
                    agent.set_property_value(c.current_task_id, 3 - current_task_id)
                self.target_waypoints[current_task_id - 1] = self.waypoints[next_waypoint_ind]
                self.target_mach_limits[current_task_id - 1] = self.mach_numbers[next_mach_ind]