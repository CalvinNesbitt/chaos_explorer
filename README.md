# chaos_explorer
Package for doing standard deterministic dynamical system's analysis in python. Inpsired by [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl)

## Roadmap

1. [ ] **Goal 0**: Have basic integrator/observer system setup.
    1. [x] Wrapped scipy integrators and coupled with observers.
    2. [ ] Create example notebooks for common workflows, including integration, observation and m-state computation.
    3. [ ] Documentation and unit tests.
2. [ ] **Goal 1**: Add lyapunov computation functionality
    1. [x] Lyapunov spectrum & CLVs when tlm provided.
    2. [ ] Jax for auto-generation of tlm.
3. [x] **Goal 2**: Add M-State computation
4. [ ] **Goal 3**: Add discrete dynamics
5. [ ] **Goal 4**: Add stochastic dynamics

