from .env_base import BaseEnv
from ..tasks import OpusTrainingTask


class OpusTrainingEnv(BaseEnv):
    """
    OpusTrainingEnv is an fly-control env for single agent with multiple missions.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None

    def load_task(self):
        #taskname = getattr(self.config, 'task', None)
        self.task = OpusTrainingTask(self.config)

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        info = {}
        return self._pack(obs), info

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading_deg = self.np_random.uniform(0., 180.)
        init_altitude_m = self.np_random.uniform(2000., 9000.)
        init_velocities_u_mps = self.np_random.uniform(120., 365.)
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading_deg,
                'ic_h_sl_ft': init_altitude_m / 0.3048,
                'ic_u_fps': init_velocities_u_mps / 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()
