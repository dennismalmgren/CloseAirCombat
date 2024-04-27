
from jsbsim_catalog import Property, JsbsimCatalog
from enum import Enum
from ..utils.utils import in_range_deg
from numpy.linalg import norm

class ExtraCatalog(Property, Enum):
    """
    A class to define and access state from Jsbsim in SI formats
    """
    accelerations_udot_m_sec2 = Property("accelerations/udot-m_sec2", "m/s²", -4.0, 4.0, access="R",
                                            update=lambda sim: sim.set_property_value(
                                                ExtraCatalog.accelerations_udot_m_sec2,
                                                sim.get_property_value(JsbsimCatalog.accelerations_udot_ft_sec2) * 0.3048
                                            )
                                            )
    accelerations_vdot_m_sec2 = Property("accelerations/vdot-m_sec2", "m/s²", -4.0, 4.0, access="R",
                                         update=lambda sim: sim.set_property_value(
                                             ExtraCatalog.accelerations_vdot_m_sec2,
                                             sim.get_property_value(JsbsimCatalog.accelerations_vdot_ft_sec2) * 0.3048
                                         )
                                         )
    accelerations_wdot_m_sec2 = Property("accelerations/wdot-m_sec2", "m/s²", -4.0, 4.0, access="R",
                                            update=lambda sim: sim.set_property_value(
                                                ExtraCatalog.accelerations_wdot_m_sec2,
                                                sim.get_property_value(JsbsimCatalog.accelerations_wdot_ft_sec2) * 0.3048
                                            )
                                            )
    position_h_agl_m = Property(
        "position/h-agl-m", "altitude above ground level [m]", -500, 26000, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.position_h_agl_m,
            sim.get_property_value(JsbsimCatalog.position_h_agl_ft) * 0.3048))
    
    position_h_sl_m = Property(
        "position/h-sl-m", "altitude above mean sea level [m]", -500, 26000, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.position_h_sl_m,
            sim.get_property_value(JsbsimCatalog.position_h_sl_ft) * 0.3048))
    
    atmosphere_crosswind_mps = Property(
        "atmosphere/crosswind-mps", "crosswind [m/s]", -100, 100, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.atmosphere_crosswind_mps,
            sim.get_property_value(JsbsimCatalog.atmosphere_crosswind_fps) * 0.3048))

    atmosphere_headwind_mps = Property(
        "atmosphere/headwind-mps", "headwind [m/s]", -100, 100, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.atmosphere_headwind_mps,
            sim.get_property_value(JsbsimCatalog.atmosphere_headwind_fps) * 0.3048))
    
    velocities_v_north_mps = Property(
        "velocities/v-north-mps", "velocity true north [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_north_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_north_fps) * 0.3048))

    velocities_v_east_mps = Property(
        "velocities/v-east-mps", "velocity east [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_east_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_east_fps) * 0.3048))

    velocities_v_down_mps = Property(
        "velocities/v-down-mps", "velocity downwards [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_down_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_down_fps) * 0.3048))

    velocities_vc_mps = Property(
        "velocities/vc-mps", "airspeed in knots [m/s]", 0, 1400, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_vc_mps,
            sim.get_property_value(JsbsimCatalog.velocities_vc_fps) * 0.3048))

    velocities_u_mps = Property(
        "velocities/u-mps", "body frame x-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_u_mps,
            sim.get_property_value(JsbsimCatalog.velocities_u_fps) * 0.3048))

    velocities_v_mps = Property(
        "velocities/v-mps", "body frame y-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_v_mps,
            sim.get_property_value(JsbsimCatalog.velocities_v_fps) * 0.3048))

    velocities_w_mps = Property(
        "velocities/w-mps", "body frame z-axis velocity [m/s]", -700, 700, access="R",
        update=lambda sim: sim.set_property_value(
            ExtraCatalog.velocities_w_mps,
            sim.get_property_value(JsbsimCatalog.velocities_w_fps) * 0.3048))

# detect functions

    detect_extreme_state = Property(
        "detect/extreme-state",
        "detect extreme rotation, velocity and altitude",
        0,
        1,
        spaces=Discrete,
        access="R",
        update=update_detect_extreme_state,
    )

    def update_detect_extreme_state(sim):
        """
        Check whether the simulation is going through excessive values before it returns NaN values.
        Store the result in detect_extreme_state property.
        """
        extreme_velocity = sim.get_property_value(JsbsimCatalog.velocities_eci_velocity_mag_fps) >= 1e10
        extreme_rotation = (
            norm(
                sim.get_property_values(
                    [
                        JsbsimCatalog.velocities_p_rad_sec,
                        JsbsimCatalog.velocities_q_rad_sec,
                        JsbsimCatalog.velocities_r_rad_sec,
                    ]
                )
            ) >= 1000
        )
        extreme_altitude = sim.get_property_value(JsbsimCatalog.position_h_sl_ft) >= 1e10
        extreme_acceleration = (
            max(
                [
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_x_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_y_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_z_norm)),
                ]
            ) > 1e1
        )  # acceleration larger than 10G
        sim.set_property_value(
            ExtraCatalog.detect_extreme_state,
            extreme_altitude or extreme_rotation or extreme_velocity or extreme_acceleration,
        )

class MissionStuff:
#Mission stuff
    def update_delta_altitude(sim):
        value = (sim.get_property_value(ExtraCatalog.target_altitude_ft) - sim.get_property_value(JsbsimCatalog.position_h_sl_ft)) * 0.3048
        sim.set_property_value(ExtraCatalog.delta_altitude, value)

    def update_delta_heading(sim):
        value = in_range_deg(
            sim.get_property_value(ExtraCatalog.target_heading_deg) - sim.get_property_value(JsbsimCatalog.attitude_psi_deg)
        )
        sim.set_property_value(ExtraCatalog.delta_heading, value)

    def update_delta_velocities(sim):
        value = (sim.get_property_value(ExtraCatalog.target_velocities_u_mps) - sim.get_property_value(ExtraCatalog.velocities_u_mps))
        sim.set_property_value(ExtraCatalog.delta_velocities_u, value)

    # @staticmethod
    # def update_property_incr(sim, discrete_prop, prop, incr_prop):
    #     value = sim.get_property_value(discrete_prop)
    #     if value == 0:
    #         pass
    #     else:
    #         if value == 1:
    #             sim.set_property_value(prop, sim.get_property_value(prop) - sim.get_property_value(incr_prop))
    #         elif value == 2:
    #             sim.set_property_value(prop, sim.get_property_value(prop) + sim.get_property_value(incr_prop))
    #         sim.set_property_value(discrete_prop, 0)

    # def update_throttle_cmd_dir(sim):
    #     ExtraCatalog.update_property_incr(
    #         sim, ExtraCatalog.throttle_cmd_dir, JsbsimCatalog.fcs_throttle_cmd_norm, ExtraCatalog.incr_throttle
    #     )

    # def update_aileron_cmd_dir(sim):
    #     ExtraCatalog.update_property_incr(
    #         sim, ExtraCatalog.aileron_cmd_dir, JsbsimCatalog.fcs_aileron_cmd_norm, ExtraCatalog.incr_aileron
    #     )

    # def update_elevator_cmd_dir(sim):
    #     ExtraCatalog.update_property_incr(
    #         sim, ExtraCatalog.elevator_cmd_dir, JsbsimCatalog.fcs_elevator_cmd_norm, ExtraCatalog.incr_elevator
    #     )

    # def update_rudder_cmd_dir(sim):
    #     ExtraCatalog.update_property_incr(
    #         sim, ExtraCatalog.rudder_cmd_dir, JsbsimCatalog.fcs_rudder_cmd_norm, ExtraCatalog.incr_rudder
    #     )

    

    # position and attitude

    delta_altitude = Property(
        "position/delta-altitude-to-target-m",
        "delta altitude to target [m]",
        -40000,
        40000,
        access="R",
        update=update_delta_altitude,
    )
    delta_heading = Property(
        "position/delta-heading-to-target-deg",
        "delta heading to target [deg]",
        -180,
        180,
        access="R",
        update=update_delta_heading,
    )
    delta_velocities_u = Property(
        "position/delta-velocities_u-to-target-mps",
        "delta velocities_u to target",
        -1400,
        1400,
        access="R",
        update=update_delta_velocities,
    )
    # controls command

    throttle_cmd_dir = Property(
        "fcs/throttle-cmd-dir",
        "direction to move the throttle",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_throttle_cmd_dir,
    )
    incr_throttle = Property("fcs/incr-throttle", "incrementation throttle", 0, 1)
    aileron_cmd_dir = Property(
        "fcs/aileron-cmd-dir",
        "direction to move the aileron",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_aileron_cmd_dir,
    )
    incr_aileron = Property("fcs/incr-aileron", "incrementation aileron", 0, 1)
    elevator_cmd_dir = Property(
        "fcs/elevator-cmd-dir",
        "direction to move the elevator",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_elevator_cmd_dir,
    )
    incr_elevator = Property("fcs/incr-elevator", "incrementation elevator", 0, 1)
    rudder_cmd_dir = Property(
        "fcs/rudder-cmd-dir",
        "direction to move the rudder",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_rudder_cmd_dir,
    )
    incr_rudder = Property("fcs/incr-rudder", "incrementation rudder", 0, 1)

    

    # Assignment variables
    # Current options:
    # 0: no mission (not used),
    # 1: travel in heading at altitude and speed
    # 2: travel to waypoint
    # 3: search area
    # 4: engage target
    # Assignments can be given in sequence (2 in a row)
    # such as task 1 = 1, task 2 = 2. 
    current_task_id = Property(
        "missions/current_task_id",
        "current_task_id",
        0,
        2,
        spaces=Discrete
    )

    task_1_type_id = Property(
        "missions/task_1_type_id",
        "task 1 type id",
        0,
        5,
        spaces=Discrete
    )

    task_2_type_id = Property(
        "missions/task_2_type_id",
        "task 2 type id",
        0,
        5,
        spaces=Discrete
    )

    #Travel in direction at altitude
    travel_1_target_position_h_sl_m = Property(
        "missions/travel-1-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    travel_1_target_attitude_psi_rad = Property(
        "missions/travel-1-target-attitude-psi-rad",
        "target heading [rad]",
        JsbsimCatalog.attitude_psi_rad.min,
        JsbsimCatalog.attitude_psi_rad.max,
    )

    travel_1_target_velocities_u_mps = Property(
        "missions/travel-1-target-velocity-u-mps",
        "target speed [mps]",
        -700,
        700
    )

    travel_1_target_time_s = Property(
        "missions/travel-1-target-time-sec",
        "target time [sec]",
        0
    )

    travel_2_target_position_h_sl_m = Property(
        "missions/travel-2-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    travel_2_target_attitude_psi_rad = Property(
        "missions/travel-2-target-attitude-psi-rad",
        "target heading [rad]",
        JsbsimCatalog.attitude_psi_rad.min,
        JsbsimCatalog.attitude_psi_rad.max,
    )

    travel_2_target_velocities_u_mps = Property(
        "missions/travel-2-target-velocity-u-mps",
        "target speed [mps]",
        -700,
        700
    )

    travel_2_target_time_s = Property(
        "missions/travel-2-target-time-sec",
        "target time [sec]",
        0
    )

    #travel to waypoint
    wp_1_1_target_position_h_sl_m = Property(
        "missions/wp-1-1-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    wp_1_1_target_position_lat_geod_rad = Property(
        "missions/wp-1-1-target-position-lat-geod-rad",
        "target geodesic latitude [rad]",
        JsbsimCatalog.position_lat_geod_rad.min,
        JsbsimCatalog.position_lat_geod_rad.max,
    )

    wp_1_1_target_position_long_gc_rad = Property(
        "missions/wp-1-1-target-position-long-gc-rad",
        "target geocentric (geodesic) longitude [rad]",
        JsbsimCatalog.position_long_gc_rad.min,
        JsbsimCatalog.position_long_gc_rad.max,
    )

    wp_1_1_target_velocities_v_north_mps = Property(
        "missions/wp-1-1-target-velocity-v-north-mps",
        "target velocity true north [mps]",
        -700,
        700
    )

    wp_1_1_target_velocities_v_east_mps = Property(
        "missions/wp-1-1-target-velocity-v-east-mps",
        "target velocity east [mps]",
        -700,
        700
    )

    wp_1_1_target_velocities_v_down_mps = Property(
        "missions/wp-1-1-target-velocity-v-down-mps",
        "target velocity downwards [mps]",
        -700,
        700
    )

    wp_1_1_target_time_s = Property(
        "missions/wp-1-1-target-time-sec",
        "target time [sec]",
        0
    )

    wp_1_2_target_position_h_sl_m = Property(
        "missions/wp-1-2-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    wp_1_2_target_position_lat_geod_rad = Property(
        "missions/wp-1-2-target-position-lat-geod-rad",
        "target geodesic latitude [rad]",
        JsbsimCatalog.position_lat_geod_rad.min,
        JsbsimCatalog.position_lat_geod_rad.max,
    )

    wp_1_2_target_position_long_gc_rad = Property(
        "missions/wp-1-2-target-position-long-gc-rad",
        "target geocentric (geodesic) longitude [rad]",
        JsbsimCatalog.position_long_gc_rad.min,
        JsbsimCatalog.position_long_gc_rad.max,
    )

    wp_1_2_target_velocities_v_north_mps = Property(
        "missions/wp-1-2-target-velocity-v-north-mps",
        "target velocity true north [mps]",
        -700,
        700
    )

    wp_1_2_target_velocities_v_east_mps = Property(
        "missions/wp-1-2-target-velocity-v-east-mps",
        "target velocity east [mps]",
        -700,
        700
    )

    wp_1_2_target_velocities_v_down_mps = Property(
        "missions/wp-1-2-target-velocity-v-down-mps",
        "target velocity downwards [mps]",
        -700,
        700
    )

    wp_1_2_target_time_s = Property(
        "missions/wp-1-2-target-time-sec",
        "target time [sec]",
        0
    )
    
    wp_2_1_target_position_h_sl_m = Property(
        "missions/wp-2-1-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    wp_2_1_target_position_lat_geod_rad = Property(
        "missions/wp-2-1-target-position-lat-geod-rad",
        "target geodesic latitude [rad]",
        JsbsimCatalog.position_lat_geod_rad.min,
        JsbsimCatalog.position_lat_geod_rad.max,
    )

    wp_2_1_target_position_long_gc_rad = Property(
        "missions/wp-2-1-target-position-long-gc-rad",
        "target geocentric (geodesic) longitude [rad]",
        JsbsimCatalog.position_long_gc_rad.min,
        JsbsimCatalog.position_long_gc_rad.max,
    )

    wp_2_1_target_velocities_v_north_mps = Property(
        "missions/wp-2-1-target-velocity-v-north-mps",
        "target velocity true north [mps]",
        -700,
        700
    )

    wp_2_1_target_velocities_v_east_mps = Property(
        "missions/wp-2-1-target-velocity-v-east-mps",
        "target velocity east [mps]",
        -700,
        700
    )

    wp_2_1_target_velocities_v_down_mps = Property(
        "missions/wp-2-1-target-velocity-v-down-mps",
        "target velocity downwards [mps]",
        -700,
        700
    )

    wp_2_1_target_time_s = Property(
        "missions/wp-2-1-target-time-sec",
        "target time [sec]",
        0
    )

    wp_2_2_target_position_h_sl_m = Property(
        "missions/wp-2-2-target-position-h-sl-m",
        "target altitude MSL [m]",
        JsbsimCatalog.position_h_sl_ft.min * 0.3048,
        JsbsimCatalog.position_h_sl_ft.max * 0.3048,
    )

    wp_2_2_target_position_lat_geod_rad = Property(
        "missions/wp-2-2-target-position-lat-geod-rad",
        "target geodesic latitude [rad]",
        JsbsimCatalog.position_lat_geod_rad.min,
        JsbsimCatalog.position_lat_geod_rad.max,
    )

    wp_2_2_target_position_long_gc_rad = Property(
        "missions/wp-2-2-target-position-lat-geod-rad",
        "target geocentric (geodesic) longitude [rad]",
        JsbsimCatalog.position_long_gc_rad.min,
        JsbsimCatalog.position_long_gc_rad.max,
    )

    wp_2_2_target_velocities_v_north_mps = Property(
        "missions/wp-2-2-target-velocity-v-north-mps",
        "target velocity true north [mps]",
        -700,
        700
    )

    wp_2_2_target_velocities_v_east_mps = Property(
        "missions/wp-2-2-target-velocity-v-east-mps",
        "target velocity east [mps]",
        -700,
        700
    )

    wp_2_2_target_velocities_v_down_mps = Property(
        "missions/wp-2-2-target-velocity-v-down-mps",
        "target velocity downwards [mps]",
        -700,
        700
    )

    wp_2_2_target_time_s = Property(
        "missions/wp-2-2-target-time-sec",
        "target time [sec]",
        0
    )

    #search area
    search_area_1_x1_grid = Property(
        "missions/search-area-1-x-1-grid",
        "search area grid x1 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_1_y1_grid = Property(
        "missions/search-area-1-y-1-grid",
        "search area grid y1 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_1_x2_grid = Property(
        "missions/search-area-1-x-2-grid",
        "search area grid x2 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_1_y2_grid = Property(
        "missions/search-area-1-y-2-grid",
        "search area grid y2 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_1_target_time_s = Property(
        "missions/search-area-1-target-time-sec",
        "target time [sec]",
        0
    )

    search_area_2_x1_grid = Property(
        "missions/search-area-2-x-1-grid",
        "search area grid x1 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_2_y1_grid = Property(
        "missions/search-area-2-y-1-grid",
        "search area grid y1 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_2_x2_grid = Property(
        "missions/search-area-2-x-2-grid",
        "search area grid x2 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_2_y2_grid = Property(
        "missions/search-area-2-y-2-grid",
        "search area grid y2 coordinate",
        0,
        1000,
        spaces=Discrete
    )

    search_area_2_target_time_s = Property(
        "missions/search-area-2-target-time-sec",
        "target time [sec]",
        0
    )

    #engage target
    akan_destroy_1_target_id = Property(
        "missions/akan-destroy-1-target-id",
        "target id",
        0,
        1000,
        spaces=Discrete
    )

    akan_destroy_1_target_time_s = Property(
        "missions/akan-destroy-1-target-time-sec",
        "target time [sec]",
        0
    )

    akan_destroy_2_target_id = Property(
        "missions/akan-destroy-2-target-id",
        "target id",
        0,
        1000,
        spaces=Discrete
    )

    akan_destroy_2_target_time_s = Property(
        "missions/akan-destroy-2-target-time-sec",
        "target time [sec]",
        0
    )

    ### OLD VARIABLES
    target_altitude_ft = Property(
        "tc/h-sl-ft",
        "target altitude MSL [ft]",
        JsbsimCatalog.position_h_sl_ft.min,
        JsbsimCatalog.position_h_sl_ft.max,
    )

    target_heading_deg = Property(
        "tc/target-heading-deg",
        "target heading [deg]",
        JsbsimCatalog.attitude_psi_deg.min,
        JsbsimCatalog.attitude_psi_deg.max,
    )

    target_velocities_u_mps = Property(
        "tc/target-velocity-u-mps",
        "target heading [mps]",
        -700,
        700
    )


    target_vg = Property("tc/target-vg", "target ground velocity [ft/s]")
    target_time = Property("tc/target-time-sec", "target time [sec]", 0)
    target_latitude_geod_deg = Property("tc/target-latitude-geod-deg", "target geocentric latitude [deg]", -90, 90)
    target_longitude_geod_deg = Property(
        "tc/target-longitude-geod-deg", "target geocentric longitude [deg]", -180, 180
    )
    heading_check_time = Property("heading_check_time", "time to check whether current time reaches heading time", 0, 1000000)
    waypoint_1_1_check_time = Property("waypoint_1_1_check_time", "time to check whether waypoint has been reached", 0, 1000000)
    waypoint_1_2_check_time = Property("waypoint_1_2_check_time", "time to check whether waypoint has been reached", 0, 1000000)
