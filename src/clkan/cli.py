from logging import getLogger
from pathlib import Path
from subprocess import run as prun
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import clkan.config as cfg

log = getLogger(__name__)


def get_git_tag() -> Optional[str]:
    try:
        return prun(
            "git rev-parse --short HEAD",
            capture_output=True,
            text=True,
            check=True,
            shell=True,
            executable="/bin/sh",
            env=None,
        ).stdout.strip()[:8]
    except Exception:
        return None


def run(config: cfg.Config, work_dir: Path):
    # Import here to speed up startup when `run` is not called
    from clkan.model import new_model
    from clkan.regression_loop import lighting_cl_loop
    from clkan.scenario import new_scenario

    git_tag = get_git_tag()
    if git_tag is not None:
        config.tags.append(git_tag)

    about_scenario, scenario = new_scenario(config.scenario)
    about_model, model = new_model(config.model, about_scenario, config.training.device)
    experiment_name = HydraConfig.get().job.override_dirname

    return lighting_cl_loop(
        config,
        model,
        about_model,
        scenario,
        about_scenario,
        work_dir,
        experiment_name,
    )


@hydra.main(version_base=None, config_path="conf", config_name="base")
def _hydra_run(config):
    config = cfg.Config.model_validate(OmegaConf.to_container(config))
    hydra_work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    try:
        values = run(config, Path(hydra_work_dir))
        log.info(values)
        return values
    except Exception as e:
        log.exception(e)
        raise e


def main():
    return _hydra_run()


if __name__ == "__main__":
    main()
