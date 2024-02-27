from typing import Optional
import random

import numpy as np
import pymap3d as pm

'''
start hårdkodat

speed mach 0.6-0.8

sub:
/namespace/new_flightpath  geopath 0 stampade geoposes

pub:
/namespace/pose     geopose = geopoint orientering
/namespace/velocity twist
/namespace/flightpath geoposes
'''

'''
geographic_msgs/msg/GeoPath
ros2 topic pub /figher_agent_cm1/new_flightpath geographic_msgs/msg/GeoPath "{poses: [{pose: {position: {latitude: 56.26628836811026, longitude: 15.264993041832879, altitude: 100.0}}}]}"

Kallinge startbana 56.26628836811026, 15.264993041832879

'''
class GeoPath:
    def __init__(self):
        self.poses = list()

class GeoPoint:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0

class GeoPose:
    def __init__(self):
        self.position = GeoPoint()
        self.orientation = Quaternion()

class Quaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

# Great source of spherical and elliptical calculations https://www.movable-type.co.uk/scripts/latlong.html

    
class CruiseMissile:

    def __init__(self, name: str, starting_location, target_path: Optional[GeoPath] = None) -> None:
        self.name = name

        self.start_path = target_path
        self.flightpath = target_path

        self.speed = 300.0  # m/s ~ mach 0.9

        self.pose = GeoPose()
        self.pose.position = GeoPoint()
        self.pose.orientation = Quaternion()
        self.pose.position.latitude = starting_location[0]
        self.pose.position.longitude = starting_location[1]
        self.pose.position.altitude = starting_location[2]

    @classmethod
    def get_geopoint(cls, lat, lon, alt):
        geo = GeoPoint()
        geo.latitude = lat
        geo.longitude = lon
        geo.altitude = alt
        return geo

    @classmethod
    def get_geopose(cls, geopose):
        position = GeoPoint()
        position.latitude = geopose.position.latitude
        position.longitude = geopose.position.longitude
        position.altitude = geopose.position.altitude
        orientation = Quaternion()
        orientation.x = geopose.orientation.x
        orientation.y = geopose.orientation.y
        orientation.z = geopose.orientation.z
        orientation.w = geopose.orientation.w
        pose = GeoPose()
        pose.position = position
        pose.orientation = orientation
        return pose

    def setup_path(self):
        # now = time.Time()  # self.get_clock().now.to_msg()
        self.flightpath = GeoPath()
        self.flightpath.header = CruiseMissile.get_header(0, 0, '0')

        poses = list()
        for p in self.start_path.poses:
            poses.append(CruiseMissile.get_geopose_stamped(0, 0, '0', p))
        self.flightpath.poses = poses

    @property
    def has_hit(self):
        return self.flightpath is None
    
    def step_path(self):
        if self.flightpath:
            if len(self.flightpath.poses) == 1:
                self.flightpath = None
            else:
                # now = time.Time()  # self.get_clock().now.to_msg()
                self.flightpath = GeoPath()

                self.start_path.poses.pop(0)
                poses = list()
                for p in self.start_path.poses:                   
                    poses.append(p)
                self.flightpath.poses = poses

    def step(self, freq=1.0):
        stepsize = 1./freq
        if self.flightpath:
            current = self.flightpath.poses[0].position
            target_lat = current.latitude
            target_long = current.longitude
            target_alt = current.altitude

            # Distance to WP from current position
            north, east, down = pm.geodetic2ned(self.pose.position.latitude, self.pose.position.longitude, self.pose.position.altitude, target_lat, target_long, target_alt)
            wp_distance = np.sqrt(north**2 + east**2 + down**2)

            if wp_distance > self.speed * stepsize:
                v_north = north / wp_distance
                v_east = east / wp_distance
                v_down = down / wp_distance

                north = north - v_north * self.speed * stepsize
                east = east - v_east * self.speed * stepsize
                if wp_distance < 50000.0:  # To preserve hight around the curving earth, keep altitude until closer to target point
                    down = down - v_down * self.speed * stepsize

                lat, lon, alt = pm.ned2geodetic(north, east, down, target_lat, target_long, target_alt)
            else:
                lat, lon, alt = target_lat, target_long, target_alt
                #a hit
                self.step_path()

            self.pose.position.latitude = lat
            self.pose.position.longitude = lon
            if wp_distance < 50000.0:  # To preserve hight around the curving earth, keep altitude until closer to target point
                self.pose.position.altitude = alt

        # If reached, call step_path and continue

linkoping = (58.40160, 15.63308, 1000.0)
jonkoping = (57.75882, 14.16009, 1000.0)
norrkoping = (58.58341, 16.23918, 1000.0)
kaliningrad = (54.73032, 20.45923, 1000.0)
stpetersburg = (59.94279, 30.24516, 1000.0)
knackpunkt = (58.25883, 27.71385, 1000.0)

cm1 = ['cruise_missile_1',
       knackpunkt,
       [linkoping]]
cm2 = ['cruise_missile_2',
       knackpunkt,
       [linkoping]]
cm3 = ['cruise_missile_3',
       knackpunkt,
       [jonkoping]]
cm4 = ['cruise_missile_4',
       knackpunkt,
       [jonkoping]]
cm5 = ['cruise_missile_5',
       knackpunkt,
       [norrkoping]]
cm6 = ['cruise_missile_6',
       knackpunkt,
       [linkoping]]
cm7 = ['cruise_missile_7',
       knackpunkt,
       [norrkoping]]
cm8 = ['cruise_missile_8',
       knackpunkt,
       [linkoping]]
cm9 = ['cruise_missile_9',
       knackpunkt,
       [norrkoping]]
cm10 = ['cruise_missile_10',
        knackpunkt,
        [jonkoping]]

cms = [cm1, cm2, cm3, cm4, cm5, cm6, cm7, cm8, cm9, cm10]


def get_geopath(points):
    flightpath = GeoPath()
    poses = list()
    for p in points:
        gps = GeoPose()
        gps.position = GeoPoint()
        gps.position.latitude = p[0]
        gps.position.longitude = p[1]
        gps.position.altitude = p[2]
        poses.append(gps)
    flightpath.poses = poses
    return flightpath


def randomize_starting_position(cm):
    cm[1] = (cm[1][0] + random.random()*0.30-0.15, cm[1][1] + random.random()*0.30-0.15, cm[1][2])


def randomize_first_turn_position(cm):
    cm[2][0] = (cm[2][0][0] + random.random()*0.30-0.15, cm[2][0][1] + random.random()*0.30-0.15, cm[2][0][2])

class BoundingRect:
    def __init__(self, lat_min, lon_min, lat_max, lon_max):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

    def contains(self, lat, lon):
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max
    
def main(args=None):
    print('Creating cruise missiles')
    boundingRect = BoundingRect(56.98033, 14.07798, 58.74902, 20.82290)

    for cm in [cms[1]]:
        randomize_starting_position(cm)
        randomize_first_turn_position(cm)
        print('Starting at Lat, Lon, Alt: ', cm[1])
        print('Targetting Lat, Lon, Alt: ', cm[2])
        cm_node = CruiseMissile(cm[0], cm[1], get_geopath(cm[2]))
        print("Lat, lon, alt: ", cm_node.pose.position.latitude, cm_node.pose.position.longitude, cm_node.pose.position.altitude)
        out_of_bounds = True
        step_id = 0
        while not cm_node.has_hit:
            cm_node.step(freq=1)
            step_id += 1
            if boundingRect.contains(cm_node.pose.position.latitude, cm_node.pose.position.longitude):
                if out_of_bounds:
                    print(f'Enter after {step_id} steps')
                    print("Lat, lon, alt: ", cm_node.pose.position.latitude, cm_node.pose.position.longitude, cm_node.pose.position.altitude)
                    out_of_bounds = False

            #print("Lat, lon, alt: ", cm_node.pose.position.latitude, cm_node.pose.position.longitude, cm_node.pose.position.altitude)

        print(f'Hit after {step_id} steps!')
        print("Lat, lon, alt: ", cm_node.pose.position.latitude, cm_node.pose.position.longitude, cm_node.pose.position.altitude)

#so the goal is to 
if __name__ == '__main__':
    main()
