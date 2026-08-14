from envs.simulators.mujoco import MujocoSimulator


SIM_TYPE_MAP: dict[str, type[MujocoSimulator]] = {
    "mujoco": MujocoSimulator,
    # "issac": IssacSimulator,
}
