from juliacall import Main as jl

class BatchedEnv:

    def __init__(self, modelfile: str, n_envs: int, n_physics_steps: int):
        jl.include("mujoco_env.jl")
        self.juliaEnv = jl.BatchedEnv(modelfile, n_envs=n_envs,
                                      n_physics_steps=n_physics_steps)

    def step(self, action):
        qpos, com = jl.step(self.juliaEnv, action)
        return {'qpos': qpos, 'com': com} #Could also be a tensordict

    @property
    def n_envs(self):
        return jl.n_envs(self.juliaEnv)
    
    @property
    def action_size(self):
        return jl.action_size(self.juliaEnv)
    
    @property
    def nthreads(self):
        return jl.Threads.nthreads()
