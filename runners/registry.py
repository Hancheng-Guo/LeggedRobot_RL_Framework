from runners.on_policy import OnPolicyRunner
# from runners.off_policy import OffPolicyRunner

RUNNER_TYPE_MAP: dict[str, type[OnPolicyRunner]] = {
    "on_policy": OnPolicyRunner,
    # "off_policy": OffPolicyRunner,
}