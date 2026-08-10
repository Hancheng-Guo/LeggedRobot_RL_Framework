from pathlib import Path
import warnings


def fill_path(
        file_name: str | None = None,
        file_dir: str | Path | None = None,
        file_path: str | Path | None = None
    ) -> Path:

    if file_path is None:
        if file_name is None:
            raise ValueError("file_name is required.")
        if file_dir is None:
            return Path() / file_name
        return Path(file_dir) / file_name
    
    if file_name is not None or file_dir is not None:
        warnings.warn(
            """
            file_path is provided,
            so that file_name and file_config_dir will be ignored.
            """
        )
        
    return Path(file_path)

def fill_paths(
        file_names: list[str] | None = None,
        file_dir: str | Path | None = None,
        file_paths: list[str | Path] | None = None
    ) -> list[Path]:

    if file_paths is None:
        if file_names is None:
            raise ValueError("file_names are required.")
        if file_dir is None:
            return [Path() / file_name for file_name in file_names]
        return [Path(file_dir) / file_name for file_name in file_names]
    
    if file_names is not None or file_dir is not None:
        warnings.warn(
            """
            file_paths are provided,
            so that file_name and file_config_dir will be ignored.
            """
        )
        
    return [Path(file_path) for file_path in file_paths]