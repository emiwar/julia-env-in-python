import MuJoCo

struct BatchedEnv
    model::MuJoCo.Model
    data::Vector{MuJoCo.Data}
    n_physics_steps::Int
end

function BatchedEnv(model::MuJoCo.Model; n_envs=1, n_physics_steps=1)
    data = [MuJoCo.init_data(model) for _=1:n_envs]
    return BatchedEnv(model, data, n_physics_steps)
end

function BatchedEnv(modelfile::String; n_envs=1, n_physics_steps=1)
    model = MuJoCo.load_model(modelfile)
    return BatchedEnv(model; n_envs, n_physics_steps)
end

n_envs(batchedEnv::BatchedEnv) = length(batchedEnv.data)
action_size(batchedEnv::BatchedEnv) = batchedEnv.model.nu

function step(batchedEnv::BatchedEnv, actions::AbstractArray)
    @assert size(actions, 1) == n_envs(batchedEnv)
    @assert size(actions, 2) == action_size(batchedEnv)
    qpos = zeros(n_envs(batchedEnv), batchedEnv.model.nq)
    com = zeros(n_envs(batchedEnv), 3)
    @Threads.threads for i=1:n_envs(batchedEnv)
        data = batchedEnv.data[i]
        data.ctrl .= view(actions, i, :)
        for _=1:batchedEnv.n_physics_steps
            MuJoCo.step!(batchedEnv.model, data)
        end
        qpos[i, :] .= data.qpos
        com[i, :] .= MuJoCo.body(data, "torso").com
    end
    return qpos, com
end