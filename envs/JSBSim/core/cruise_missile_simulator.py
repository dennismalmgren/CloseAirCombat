from .simulator import BaseSimulator, AircraftSimulator
import numpy as np
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA
from collections import deque

class CruiseMissileSimulator(BaseSimulator):
    #Hit indicates that the missile has passed through the area
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2
    SHOTDOWN = 3

    #geodetic: (lontitude, latitude, altitude)
    @classmethod
    def create(cls, 
               launch_geodetic: np.ndarray, 
               target_geodetic: np.ndarray,
               origin: np.ndarray, #lat0, lon0, alt0
               uid: str, 
               march_speed: float, 
               dt: float):
        missile = CruiseMissileSimulator("B0100", color="Blue", march_speed=march_speed, dt=dt)
        missile.launch(launch_geodetic, target_geodetic, origin)
        return missile

    def __init__(self,
                 uid="B0100",
                 color="Blue",
                 march_speed=250,
                 dt=1 / 12.0):
        super().__init__(uid, color, dt)
        self.__status = CruiseMissileSimulator.INACTIVE
        self.march_speed = march_speed
        self.model = "F16"
        # missile parameters (for AIM-9L)
        self._g = 9.81      # gravitational acceleration
        self._t_max = 60*5    # time limitation of missile life
        self._t_thrust = 3  # time limitation of engine
        self._Isp = 120     # average specific impulse
        self._Length = 2.87
        self._Diameter = 0.127
        self._cD = 0.4      # aerodynamic drag factor
        self._m0 = 84       # mass, unit: kg
        self._dm = 6        # mass loss rate, unit: kg/s
        self._K = 3         # proportionality constant of proportional navigation
        self._nyz_max = 30  # max overload
        self._Rc = 300      # radius of explosion, unit: m
        self._v_min = 150   # minimun velocity, unit: m/s
        self.render_explosion = False

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.__status == CruiseMissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target"""
        return self.__status == CruiseMissileSimulator.HIT

    @property
    def is_done(self):
        """Missile is already exploded"""
        return self.__status == CruiseMissileSimulator.HIT \
            or self.__status == CruiseMissileSimulator.MISS \
                or self.__status == CruiseMissileSimulator.SHOTDOWN
        
    def shotdown(self):
        self.__status = CruiseMissileSimulator.SHOTDOWN
    
    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)


    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, launch_geodetic: np.ndarray, 
               target_geodetic: np.ndarray, 
               origin: np.ndarray):
        self._target_position = np.zeros(3)
        self._target_position[:] = LLA2NEU(*target_geodetic, *origin)
        self._geodetic[:] = launch_geodetic
        self._position[:] = LLA2NEU(*self._geodetic, *origin)
        #find the directional vector north/east
        dir = (self._target_position - self._position) / (np.linalg.norm(self._target_position - self._position))
        """(v_north, v_east, v_up), unit: m/s"""
        self._velocity[:] = np.array(self.march_speed * dir, dtype=np.float32)
        """(roll, pitch, yaw), unit: rad"""
        self._posture[:] = np.array([0.0, 0.0, 0.0])
        self._posture[2] = np.arctan2(self._velocity[1], self._velocity[0]) #assuming this is correct
        self.lon0, self.lat0, self.alt0 = origin

        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = CruiseMissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit
        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc:
            self.__status = CruiseMissileSimulator.HIT
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen:
            self.__status = CruiseMissileSimulator.MISS
        else:
            self._state_trans(action)

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        pass

    def _guidance(self):
        """
        Guidance law, proportional navigation
        """
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m)
        x_t, y_t, z_t = self._target_position
        dx_t, dy_t, dz_t = np.zeros(3)
        #x_t, y_t, z_t = self.target_aircraft.get_position()
#        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """
        State transition function
        """
        # update position & geodetic
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        # update velocity & posture
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v**2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
        self._dtheta = self._g / v * (nz - np.cos(theta))
        v = self.march_speed
#        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])
        # update mass
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm

class AntiCruiseMissileSimulator(BaseSimulator):

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: CruiseMissileSimulator, uid: str, missile_model: str = "AIM-9L"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = AntiCruiseMissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = AntiCruiseMissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_missile = None  # type: CruiseMissileSimulator
        self.render_explosion = False

        # missile parameters (for AIM-9L)
        self._g = 9.81      # gravitational acceleration
        self._t_max = 60    # time limitation of missile life
        self._t_thrust = 3  # time limitation of engine
        self._Isp = 120     # average specific impulse
        self._Length = 2.87
        self._Diameter = 0.127
        self._cD = 0.4      # aerodynamic drag factor
        self._m0 = 84       # mass, unit: kg
        self._dm = 6        # mass loss rate, unit: kg/s
        self._K = 3         # proportionality constant of proportional navigation
        self._nyz_max = 30  # max overload
        self._Rc = 300      # radius of explosion, unit: m
        self._v_min = 150   # minimun velocity, unit: m/s
        

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.__status == AntiCruiseMissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target"""
        return self.__status == AntiCruiseMissileSimulator.HIT

    @property
    def is_done(self):
        """Missile is already exploded"""
        return self.__status == AntiCruiseMissileSimulator.HIT \
            or self.__status == AntiCruiseMissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # exact expression (Reference: https://www.cnblogs.com/pathjh/p/9127352.html)


    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_missile.get_position() - self.get_position())
 
    def launch(self, parent: AircraftSimulator):
        # inherit kinetic parameters from parent aricraft
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # missile's roll remains zero
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = AntiCruiseMissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit
        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def target(self, target: CruiseMissileSimulator):
        self.target_missile = target  # TODO: change target?
        #self.target_missile.under_missiles.append(self)

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_missile.is_alive:
            self.__status = AntiCruiseMissileSimulator.HIT
            self.target_missile.shotdown()
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen or not self.target_missile.is_alive:
            self.__status = AntiCruiseMissileSimulator.MISS
        else:
            self._state_trans(action)

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_missile = None

    def _guidance(self):
        """
        Guidance law, proportional navigation
        """
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m)
        x_t, y_t, z_t = self.target_missile.get_position()
        dx_t, dy_t, dz_t = self.target_missile.get_velocity()
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """
        State transition function
        """
        # update position & geodetic
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        # update velocity & posture
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v**2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
        self._dtheta = self._g / v * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])
        # update mass
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm
