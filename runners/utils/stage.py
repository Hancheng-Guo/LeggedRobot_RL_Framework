from pathlib import Path

from utils.config import load_yaml, get_yaml_value


class StageManager:

    def __init__(
        self,
        config_name: str,
    ) -> None:
        self.config = load_yaml(
            Path(get_yaml_value("configs/base.yaml", "stage_manager_config_dir"))
            / config_name
        )


    # def