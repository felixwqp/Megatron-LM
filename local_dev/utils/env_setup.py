import os
import logging

def set_log(module_name: str):
    # Get the logger for the specific module
    logger = logging.getLogger(module_name)
    

    # Set the level to DEBUG for this module only
    logger.setLevel(logging.DEBUG)

    # Configure basic logging settings
    logging.basicConfig(
        level=logging.INFO,  # Default for other modules
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def set_print_env():
    print("MASTER_ADDR: ", os.environ['MASTER_ADDR'])
    print("MASTER_PORT: ", os.environ['MASTER_PORT'])
    print("WORLD_SIZE: ", os.environ['WORLD_SIZE'])
    print("RANK: ", os.environ['RANK'])
    print("PYTHONLOGLEVEL", os.getenv("PYTHONLOGLEVEL"))

    # NCCL log level
    os.environ["NCCL_DEBUG"]="DEBUG"
    os.environ["NCCL_COMM_ID"]="127.0.0.1:29501" # Override to a addr not used.

    # Python log level
    os.environ["PYTHONLOGLEVEL"]="DEBUG" # invalid

    # PyTorch log level
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    
    set_log("megatron.core.dist_checkpointing.strategies.filesystem_async")
    set_log("megatron.core.dist_checkpointing.strategies.state_dict_saver")