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
        self.min_waypoint_distance = 1000 #m
        self.min_waypoint_altitude = 1000 #m 
        self.max_waypoint_altitude = 10000 #m
        self.min_north = 0 #m
        self.max_north = 200_000 #m
        self.min_east = 0 #m
        self.max_east = 400_000 #m
        self.min_speed_x = 100  #m/s
        self.max_speed_x = 450  #m/s
        self.min_speed_y = 100  #m/s
        self.max_speed_y = 450  #m/s
        self.min_speed_z = 0  #m/s
        self.max_speed_z = 50  #m/s


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
    
    def create_random_waypoint(self, agent, env):
        agent.set_property_value(c.current_task_id, 1) #1 or 2. we always go with 0 for now..
        agent.set_property_value(c.task_1_type_id, 2) #1 for heading, 2 for waypoint.
        agent.set_property_value(c.task_2_type_id, 0) #0 for no mission.

        current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.

        current_altitude = agent.get_property_value(c.position_h_sl_m)
        current_lat_deg = agent.get_property_value(c.position_lat_geod_deg)
        current_lon_deg = agent.get_property_value(c.position_long_gc_deg)

        current_location_ned = LLA2NED(current_lat_deg, current_lon_deg, current_altitude, env.task.lat0, env.task.lon0, env.task.alt0)
        current_location_neu = np.array([current_location_ned[0], current_location_ned[1], -current_location_ned[2]])

        waypoint_north = env.np_random.uniform(self.min_north, self.max_north)
        waypoint_east = env.np_random.uniform(self.min_east, self.max_east)
        waypoint_altitude = env.np_random.uniform(self.min_waypoint_altitude, self.max_waypoint_altitude)
        
        flightmode_choice = env.np_random.integers(0, 3)
        if flightmode_choice == 0:
            flightmode = FlightModes.LOITER
        elif flightmode_choice == 1:
            flightmode = FlightModes.CRUISE
        else:
            flightmode = FlightModes.ENGAGE
    
        target_location_neu = np.array([waypoint_north, waypoint_east, waypoint_altitude])
        target_velocity = np.array([env.np_random.uniform(self.min_speed_x, self.max_speed_x), env.np_random.uniform(self.min_speed_y, self.max_speed_y), env.np_random.uniform(self.min_speed_z, self.max_speed_z)])

        #target_location_neu = np.array([100000.0,      0.0,   4000.0])
        #target_velocity = np.array([     0.0,    200.0,    100.0])
        #current_location_neu = np.array([     0.0,      0.0,   10000.0])

        wp = Waypoint(target_location_neu, target_velocity)
        fd = FlightDirective(current_location_neu, flightmode, wp)
        p_and_t = fd.path_and_time()
        wp0 = p_and_t[0]
        wp0_ned = wp0[:3] #actually neu
        wp0_ned[2] = -wp0_ned[2]
        wp0_ned[0], wp0_ned[1], wp0_ned[2] = NED2LLA(wp0_ned[0], wp0_ned[1], wp0_ned[2], env.task.lat0, env.task.lon0, env.task.alt0)
        agent.set_property_value(c.wp_1_1_target_position_h_sl_m, wp0_ned[2])
        agent.set_property_value(c.wp_1_1_target_position_lat_geod_rad, wp0_ned[0] * np.pi / 180)
        agent.set_property_value(c.wp_1_1_target_position_long_gc_rad, wp0_ned[1] * np.pi / 180)
        agent.set_property_value(c.wp_1_1_target_velocities_v_north_mps, wp0[3])
        agent.set_property_value(c.wp_1_1_target_velocities_v_east_mps, wp0[4])
        agent.set_property_value(c.wp_1_1_target_velocities_v_down_mps, -wp0[5])            
        agent.set_property_value(c.wp_1_1_target_time_s, current_time + wp0[6])
        wp1 = p_and_t[1]
        wp1_ned = wp1[:3] #actually neu
        wp1_ned[2] = -wp1_ned[2]
        wp1_ned[0], wp1_ned[1], wp1_ned[2] = NED2LLA(wp1_ned[0], wp1_ned[1], wp1_ned[2], env.task.lat0, env.task.lon0, env.task.alt0)
        agent.set_property_value(c.wp_1_2_target_position_h_sl_m, wp1_ned[2])
        agent.set_property_value(c.wp_1_2_target_position_lat_geod_rad, wp1_ned[0] * np.pi / 180)
        agent.set_property_value(c.wp_1_2_target_position_long_gc_rad, wp1_ned[1] * np.pi / 180)
        agent.set_property_value(c.wp_1_2_target_velocities_v_north_mps, wp1[3])
        agent.set_property_value(c.wp_1_2_target_velocities_v_east_mps, wp1[4])
        agent.set_property_value(c.wp_1_2_target_velocities_v_down_mps, -wp1[5])
        agent.set_property_value(c.wp_1_2_target_time_s, current_time + wp1[6])
    
    def reset(self, env):
        for agent_id in env.agents:
            agent = env.agents[agent_id]
            self.create_random_waypoint(agent, env)
            
            
            #To reach a waypoint, this implementation
            # 1) generates a "heading" or rather, 
            #    intermediate waypoint (x, y, z), together with 
            #    expected speed on arrival (vx, vy, vz)
            #    together with expected time to arrive at the intermediate waypoint.
            # 2) The total time to arrive at the final waypoint, which is a curving at the end.
            # we therefore want to train our algorithm at 
            # 1) "seeing" the delta w.r.t the two phases and 
            # 2) rewarding planar flight during the first phase, followed by low jerk entry during the second phase.
            # then we add the observations necessary for the other tasks.

            #A training scenario is therefore
            # introduce a 'current task' parameter
            # add two waypoints as 'missions'. 
            # 1) single waypoint
            # this is a follow heading-task followed by meet/hit a location at velocity constraints task.
            #how do I represent those?
            
            


#   c.position_h_sl_m,                  # 0. altitude  (unit: m)
#            c.position_lat_geod_deg,            # 1. latitude geodetic (unit: deg)
#            c.position_long_gc_deg,             # 2. longitude geocentric (same as geodetic) (unit: deg)
#            current_north = agent.get_property_value(c.position_north_m)
#            current_east = agent.get_property_value(c.position_east_m)
#            current_down = agent.get_property_value(c.position_down_m)
#            current_speed_x = agent.get_property_value(c.velocities_v_north_mps)
#            current_speed_y = agent.get_property_value(c.velocities_v_east_mps)
#            current_speed_z = agent.get_property_value(c.velocities_v_down_mps)
#            current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.
#            current_heading_rad = agent.get_property_value(c.attitude_heading_true_rad) 
#            current_speed = agent.get_property_value(c.velocities_u_mps) 
#            current_time = agent.get_property_value(c.simulation_sim_time_sec) #will be at least.
            #also: set task values so that they can be returned by reset.
 #           agent.set_property_value(c.current_task_id, 0) #0 or 1. we always go with 0 for now..
#            agent.set_property_value(c.task_1_type_id, 1) #1 for heading, 2 for waypoint.
#            agent.set_property_value(c.task_2_type_id, 0) #0 for no mission.
 #           agent.set_property_value(c.travel_1_target_position_h_sl_m, current_altitude)
 #           agent.set_property_value(c.travel_1_target_attitude_psi_rad, current_heading_rad)
 #           agent.set_property_value(c.travel_1_target_velocities_u_mps, current_speed)
 #           agent.set_property_value(c.heading_check_time, (self.check_interval + current_time))    
 #           agent.set_property_value(c.travel_1_target_time_s, (self.check_interval + current_time))
            
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
        current_time = agent.get_property_value(c.simulation_sim_time_sec)
        #waypoint_1_time = agent.get_property_value(c.wp_1_1_target_time_s)
        waypoint_2_time = agent.get_property_value(c.wp_1_2_target_time_s)

        #check time is initially 0. This task works because the agent was initialized with a delta heading of 0 (target heading == current heading)
        # check heading when simulation_time exceed check_time

        if current_time >= waypoint_2_time:
            self.create_random_waypoint(agent, env)

    