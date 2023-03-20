ACCELERATE_CONFIG=$1
JOB_CONFIG=$2
SCRIPT_PATH=$3
export NCCL_DEBUG=INFO
echo $BEAKER_REPLICA_RANK
export PYTHONPATH='/stage/code/:$PYTHONPATH'
accelerate launch --config_file $ACCELERATE_CONFIG --machine_rank $BEAKER_REPLICA_RANK --main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME --main_process_port 29401 $SCRIPT_PATH --experiment_name test --log_to_wandb --entity_name nlp-gym --base_path_to_store_results /output --config_path $JOB_CONFIG