# MNIST Example

This example demonstrates how to integrate STADLE into MNIST learning code using both PyTorch.

## Setup Aggregator

0. Purchase STADLE license.

1. Sign up, login, and create a project in [stadle.ai](https://www.stadle.ai/).

2. Create a Project. At least one aggregator must already be running on a project.


## Execution

Go to `basicClientVer` directory and follow the steps below.
Please refer to [Usage](https://stadle-documentation.readthedocs.io/en/latest/usage.html) for more details on the server-side components and the integration of the STADLE `BasicClient`.

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

    ```bash
    stadle upload-model --config_path client_config.json
    ```

    The configuration file `client_config.json` must be modified to correspond with the server-side components. 

2. Run the FL Client.

    ```bash
    python mnist_ml_agent.py  --agent_name <AGENT_NAME> --num_rounds <NUM_ROUNDS>
    ```

    If multiple (non-admin) agents are being created on the same machine, make sure to specify unique agent names using the `--agent_name` flag when running `mnist_ml_agent.py` or the `agent_name` parameter in the `BasicClient` constructor.

