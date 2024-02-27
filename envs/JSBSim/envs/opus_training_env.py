from .env_base import BaseEnv
from ..tasks import OpusTrainingTask
from ..curricula import OpusCurriculum, OpusCurriculumWaypoints
from ..core.catalog import Catalog as c

class OpusTrainingEnv(BaseEnv):
    """
    OpusTrainingEnv is an fly-control env for single agent with multiple missions.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        
        self._init_states = None

    #set by curriculum.
    def load_curriculum(self):
        #self.curriculum = OpusCurriculumWaypoints(self.config)
        self.curriculum = OpusCurriculum(self.config)

    def load_task(self):
        #taskname = getattr(self.config, 'task', None)
        self.task = OpusTrainingTask(self.config)

    def reset(self):
        self.current_step = 0
        init_states = self.curriculum.create_init_states(self)
        self.reset_simulators(init_states)
        self.task.reset(self)
        self.curriculum.reset(self)
        obs = self.get_obs()
        info = {}
        return self._pack(obs), info

    def reset_simulators(self, init_states):
        if self._init_states is None:
            self._init_states = [sim.init_state.copy() for sim in self.agents.values()]

        for state_ind, agent_id in enumerate(self.agents.keys()):
            new_init_state = init_states[agent_id]
            self._init_states[state_ind].update(new_init_state)

        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self._init_states[idx])
        self._tempsims.clear()
