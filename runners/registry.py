from runners.base import BaseRunner
from runners.on_policy import OnPolicyRunner
# from runners.off_policy import OffPolicyRunner

RUNNER_TYPE_MAP: dict[str, BaseRunner] = {
    "on_policy": OnPolicyRunner,
    # "off_policy": OffPolicyRunner,
}