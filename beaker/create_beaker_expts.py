from argparse import ArgumentParser
from beaker import Beaker
from beaker.data_model import (ExperimentSpec, TaskSpec, ImageSource,
                               TaskContext, ResultSpec, TaskResources,
                               EnvVar, DataMount, DataSource)
from os import PathLike
from typing import Union
from nlp_gym.core_components.sweep import split_config, dict_hash
import yaml
from tempfile import TemporaryDirectory
import os
from datetime import datetime


def create_dataset(beaker: Beaker, dataset_name: str, source_path: Union[str, PathLike]):
    info = beaker.dataset.create(dataset_name, source_path, force=True)
    return info.full_name


def create_experiment(config_path: str,
                      accelerate_config_path: str,
                      workspace: str,
                      image_name: str,
                      cluster_name: str,
                      priority: str,
                      gpu_count: int,
                      output_path: str,
                      project_name: str,
                      log_to_wandb: bool,
                      n_runs: int,
                      entity_name: str):

    # load the config
    with open(config_path, "r") as fp:
        sweep_config = yaml.safe_load(fp)

    # expand configs
    expanded_configs = split_config(sweep_config)

    for config_ix, config in enumerate(expanded_configs):
        for run in range(n_runs):
            config["run"] = run
            config["project_name"] = project_name
            config["created_at"] = str(datetime.now())

            with TemporaryDirectory() as temp_dir:
                config_name = config_path.split("/")[-1]
                config_path_ = os.path.join(temp_dir, config_name)

                # dump the individual file to the temp file
                with open(config_path_, "w") as fp:
                    yaml.dump(config, fp)

                # beaker instance
                beaker = Beaker.from_env(default_workspace=workspace)

                # create a dataset for the config file dynamically and get the ID
                experiment_id = dict_hash(config)
                dataset_name = f"{experiment_id}_configs"
                ds_config_full_name = create_dataset(
                    beaker, dataset_name, config_path_)

                # build command list
                command_list = ["accelerate launch --config_file accelerate_config.yaml --machine_rank $BEAKER_REPLICA_RANK --main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME --num_machines 1 --num_processes 8",
                                "scripts/training/train_text_generation.py",
                                "--project_name", project_name,
                                "--experiment_name", experiment_id,
                                "--base_path_to_store_results", output_path]
                if entity_name is not None:
                    command_list.append("--entity_name")
                    command_list.append(entity_name)

                if log_to_wandb:
                    command_list.append("--log_to_wandb")

                # append config path to cmd
                command_list.append("--config_path")
                command_list.append(f"/data/{config_name}")

                # beaker spec
                spec = ExperimentSpec(tasks=[
                    TaskSpec(name=experiment_id,
                             image=ImageSource(beaker=image_name),
                             context=TaskContext(
                                 cluster=cluster_name, priority=priority),
                             resources=TaskResources(gpu_count=gpu_count),
                             result=ResultSpec(path=output_path),
                             env_vars=[EnvVar(name="WANDB_API_KEY",
                                              secret="WANDB_API_KEY")],
                             datasets=[DataMount(mount_path="/data",
                                                 source=DataSource(beaker=ds_config_full_name))],
                             command=command_list,
                             )
                ],
                    description=project_name)

                # create experiment
                exp_info = beaker.experiment.create(name=experiment_id,
                                                    spec=spec,
                                                    workspace=workspace)
                print(exp_info)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create beaker experiments on cluster")
    parser.add_argument("--config_path", type=str,
                        help="path to the config file")
    parser.add_argument("--accelerate_config_path", type=str,
                        help="path to the config file")
    parser.add_argument("--n_runs", type=int,
                        help="Number of seeds to run", default=1)
    parser.add_argument("--workspace", type=str,
                        default="ai2/nlpgym2.0")
    parser.add_argument("--image_name", type=str,
                        default="rajkumarrrk/nlpgym2.0")
    parser.add_argument("--cluster_name", type=str,
                        default="ai2/mosaic-cirrascale")
    parser.add_argument("--priority", type=str,
                        default="preemptible")
    parser.add_argument("--project_name", type=str,
                        help="Project name",
                        default="nlp_gym_exps")
    parser.add_argument("--entity_name", type=str,
                        help="Entity name",
                        default="nlp-gym")
    parser.add_argument("--gpu_count", type=int,
                        help="Number of GPUs",
                        default=2)
    parser.add_argument("--output_path",
                        type=str, help="Path to save results",
                        default="/output")
    parser.add_argument("--log_to_wandb", action="store_true",
                        help="Whether to use wandb logging", default=True)
    args = parser.parse_args()
    create_experiment(args.config_path, args.accelerate_config_path,
                      args.workspace, args.image_name,
                      args.cluster_name, args.priority, args.gpu_count,
                      args.output_path, args.project_name, args.log_to_wandb,
                      args.n_runs, args.entity_name)
