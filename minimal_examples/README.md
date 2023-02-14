# Minimal Examples

These examples demonstrate how to integrate STADLE into minimal deep learning code using both PyTorch and Keras.

In order to run any of the STADLE examples, at least one aggregator must already be running.  In addition, the configuration files located in the `config` directory must be modified to correspond with the server-side components if running on multiple instances.  If multiple (non-admin) agents are being created on the same machine, make sure to specify unique agent names using the `--agent_name` flag when running `minimal_fl_agent.py` or the `agent_name` parameter in the `BasicClient` constructor.

Please refer to [Usage](https://stadle-documentation.readthedocs.io/en/latest/usage.html) for more details on the server-side components and the integration of the STADLE `BasicClient`.

## Execution

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

    ```bash
    stadle upload-model --config_path client_config.json
    ```

2. Run the FL Client.

    ```bash
    python minimal_fl_agent.py  --agent_name <AGENT_NAME> --num_rounds <NUM_ROUNDS>
    ```
