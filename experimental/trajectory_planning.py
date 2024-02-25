from typing import List, Dict, Any
from enum import IntEnum, auto

import numpy as np


class FlightModes(IntEnum):
    LOITER = auto()
    CRUISE = auto()
    ENGAGE = auto()


# F-16A (AFG 2, Vol-1, Addn 58)
F16_FLIGHT_DATA = {
    # data tabulated at mach numbers
    "mach": [
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2
    ],
    # estimate speed [m/s] of Mach 1 at altitudes
    "mach_1": [
        340.3, 334.4, 322.3, 309.7
    ],
    # data tabulated at altitudes
    "altitude": [
        0.0, 1524.0, 4572.0, 7620.0
    ],
    # maximum g load
    "max_g": 5.0,
    # minimum turn rate at altitude and speed given aerodynamic and g-load limitations 
    # turn_rate[speed_n][altitude_n] degrees per second
    "turn_rate": [
        [18.5, 17.0, 13.5, 12.0, 10.5,  9.0,  8.0,  7.5,  7.0],
        [16.0, 17.0, 15.0, 12.0, 10.5,  9.5,  8.5,  7.5,  7.0],
        [11.0, 13.5, 14.5, 12.5, 11.0,  9.5,  9.0,  8.0,  7.0],
        [ 7.0,  9.0, 11.0, 12.5, 11.0, 10.0,  9.0,  8.0,  7.5]
    ],
    # cruise speed for sustained climb at 
    "max_climb": 50.8,  # meter per second
    "cruise_speed": {
        FlightModes.LOITER: 0.6,
        FlightModes.CRUISE: 0.9,
        FlightModes.ENGAGE: 1.2        
    }
}


class FlightProperties:
    """
    Stores flight properties of platform.
    Linear interpolation interface.
    """
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def cruise_speed(self, mode: FlightModes, pitch: float = 0.0, altitude: float = 0.0) -> float:
        cruise = self.data["cruise_speed"][mode]*np.interp(
            altitude, self.data["altitude"], self.data["mach_1"]
        )
        if pitch <= 0.0:
            return cruise        
        
        climb = cruise*np.sin(pitch)
        if climb <= self.data["max_climb"]:
            return cruise
        
        return self.data["max_climb"]/np.sin(pitch)

    def min_turn(self, mach: float, altitude: float = 0.0):
        min_turn_rate = np.interp(
            x=altitude,
            xp=self.data["altitude"],
            fp=[
                np.interp(
                    x=mach,
                    xp=self.data["mach"],
                    fp=self.data["turn_rate"][i]
                )
                for i in range(len(self.data["altitude"]))
            ]
        )
        # turn rate in degrees per second, compute radius
        return mach*self.mach_at_altitude(altitude=altitude)/(min_turn_rate*np.pi/180.0)

    def mach_at_altitude(self, altitude: float = 0.0) -> float:
        return np.interp(
            x=altitude,
            xp=self.data["altitude"],
            fp=self.data["mach_1"]
        )


FLIGHT_PROPERTIES = FlightProperties(F16_FLIGHT_DATA)


class Waypoint:
    def __init__(self, location: np.ndarray, velocity: np.ndarray):
        self.location = location
        self.velocity = velocity
        self.speed = np.linalg.norm(velocity)
        self.direction = velocity/self.speed

        assert location[2] >= 0.0

    def approach_heading_from_point(self, point: np.ndarray, turn_radius: float) -> tuple[np.ndarray, float]:
        """
        Computes a point on the surface of a horn torus around the velocity axis and
        centered at the waypoint location such that a flying platform that passes
        through the waypoint from the platforms given starting point is guaranteed
        to be able to turn along the surface of the torus and achieve the waypoint's
        velocity criteria at the waypoints location.

        This method only works if the starting point is far from the torus, or if
        the starting heading of the platform is close to the desired heading by the
        routing solution.

        Returns the heading point and the required turn angle to achieve the target.
            state.
        Returns None, 0.0 if the starting position is inside the torus.

        For flying purposes it is desirable to aim slightly astern of the given
        waypoint, this will maintain flight path guarantees while allowing the model
        to continously produce a valid solution.
        """
        aim_direction = self.location - point
        r1 = self.direction
        # orthonormal basis for manuvering plane: r1, r2
        dir_projected_distance = np.dot(aim_direction, r1)
        # one step of Grahm Schmidth
        r2 = aim_direction - r1*dir_projected_distance
        r2_norm = np.linalg.norm(r2)

        # check whether the point is on the target velocity axis
        if np.isclose(r2_norm, 0.0):
            if dir_projected_distance >= 0.0:
                # if behind the target, it may be targeted directly
                return self.location, 0.0
            else:
                # else, any vector perpendicular to the target velocity works
                # create one by taking the cross product with a vector not in its span.
                addir = np.array([0.0, 0.0, 1.0])  # try upwards
                if np.allclose(r1 - r1*np.dot(addir, r1), np.array([0.0, 0.0, 0.0])):
                    addir = np.array([1.0, 0.0, 0.0])  # otherwise this will work
                
                # one step of Grahm Schmidth
                r2 = np.cross(addir, r1)
                r2 = r2 - r1*np.dot(r2, r1)
                r2 /= np.linalg.norm(r2)
        else:
            r2 /= r2_norm

        T = np.array([r1, r2])  # transformation to manuvering plane (R3 -> R2)
        T_inv = np.transpose(T)  # inverse transformation (to subspace)

        p_loc = np.matmul(T, -aim_direction)  # transformed starting point
        rad_offs = np.array([0, turn_radius])  # offset of circle
        transl_p = p_loc + rad_offs  # translated starting point (centering circle)
        # distance from starting point to center of circle
        sdist_to_center = np.dot(transl_p, transl_p)

        # lower tangent point on circle
        target = ((turn_radius**2/sdist_to_center)*transl_p
                  - (np.sqrt(sdist_to_center - turn_radius**2)*turn_radius/sdist_to_center)
                    *np.array([-transl_p[1], transl_p[0]]))
        
        turn_angle = np.arccos(target[1]/turn_radius)
        if target[0] > 0:
            turn_angle = 2*np.pi - turn_angle
        
        target -= rad_offs

        # return inverse transform
        return np.matmul(T_inv, target) + self.location, turn_angle

    def __str__(self):
        return f"<Waypoint: x={self.location}, v={self.velocity}>"


class FlightDirective:
    def __init__(self, start: np.ndarray, mode: FlightModes, *waypoints: List[Waypoint]):
        self.start = start
        self.mode = mode
        self.waypoints = waypoints

    def _time_of_segment(self, start: np.ndarray, waypoint: Waypoint):
        """
        Computes the waypoint and time of a line-curve segment in the flight plan.
        """
        altitude = waypoint.location[2]
        mach_at_turn = waypoint.speed/FLIGHT_PROPERTIES.mach_at_altitude(altitude=altitude)
        turn_radius = FLIGHT_PROPERTIES.min_turn(mach=mach_at_turn, altitude=altitude)
        heading, turn = waypoint.approach_heading_from_point(start, turn_radius=turn_radius)
        
        travel = heading - start
        average_altitude = (heading[2] + start[2])/2
        distance = np.linalg.norm(travel)
        direction = travel/distance
        climb = np.pi/2 - np.arccos(direction[2])
        cruise = FLIGHT_PROPERTIES.cruise_speed(mode=self.mode, pitch=climb, altitude=average_altitude)

        turn_distance = turn*turn_radius
        #print(f"Turn angle: {turn*180/np.pi}, turn distance: {turn_distance}, cruise distance: {distance}")
        #print(f"Turn speed: {wp.speed}, turn time: {turn_distance/waypoint.speed} cruise speed: {cruise}, cruise time: {distance/cruise}")
        cruise_time = distance/cruise
        return np.concatenate((heading, direction*waypoint.speed)), cruise_time, turn_distance/waypoint.speed + cruise_time

    def path_and_time(self) -> List[tuple[np.ndarray, float]]:
        """
        Estimates the total time of flying the flight path based on platform properties.
        A list of waypoints (x, y, z, vx, vy, vz, t) with times relative to the starting
        time of the manouver.
        """
        start = self.start
        headings = []
        cumulative = 0.0
        for wp in self.waypoints:
            heading, time_at_heading, total_time = self._time_of_segment(start[0:3], wp)
            #This seems wrong.
            heading = np.concatenate((heading, [cumulative + time_at_heading]))
            headings.append(heading)
            cumulative += total_time
            start = np.concatenate((wp.location, wp.velocity, [cumulative]))
            headings.append(start)
        return headings


if __name__ == "__main__":
    target_location_neu = np.array([100000.0,      0.0,   4000.0])
    target_velocity = np.array([     0.0,    200.0,    100.0])
    current_location_neu = np.array([     0.0,      0.0,   10000.0])

    wp = Waypoint(target_location_neu, target_velocity)
    fd = FlightDirective(current_location_neu, FlightModes.CRUISE, wp)

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
    
    p_and_t = fd.path_and_time()
    print(len(p_and_t))
    print(fd.path_and_time())