import time
import numpy as np
import python_interface

model_file = "rodent_with_floor_scale080_edits.xml"

print("Loding the model...")
env = python_interface.BatchedEnv(model_file, n_envs=512, n_physics_steps=5)

#Julia JIT-compiles, so we shouldn't count the first call into the
#benchmark
print("Call the step the first time...")
action = np.random.normal(0, 0.2, (env.n_envs, env.action_size))
obs = env.step(action)

print("Starting benchmark...")
n_steps_per_env = 100
start_time = time.time()
for step in range(n_steps_per_env):
    action = np.random.normal(0, 0.2, (env.n_envs, env.action_size))
    obs = env.step(action)
    #print(obs["com"])
end_time = time.time()
tot_time = end_time - start_time
tot_steps = n_steps_per_env * env.n_envs
steps_per_second = tot_steps / tot_time
n_threads = env.nthreads
sps_per_thread = steps_per_second / n_threads
print(f"It took {tot_time:.2f}s to run {tot_steps} steps using {n_threads} threads.")
print(f"That is {steps_per_second:.2f} step/s in total, and {sps_per_thread:.2f} per thread.")
