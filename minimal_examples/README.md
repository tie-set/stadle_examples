# Minimal Examples

This guide provides examples of how to integrate STADLE with basic deep learning code using PyTorch and Keras frameworks.

## Configuration

### Aggregator IP

The `client_config.json` files in the `pytorch` and `keras` directories are updated to match the server-side.
In particluar, set the `aggr_ip` in `client_config.json` to the `LB Address to Connect` displayed on the STADLE Dashboard page.

### Agent Names

When running multiple agents on the same machine, assign a unique name to each agent. This can be achieved by:

- Using the `--agent_name` flag with `minimal_fl_agent.py`.
- Setting the `agent_name` parameter in the `BasicClient` constructor in your code.

## Execution

Change the directory to `pytorch` or `keras` first.

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

```bash
 stadle upload-model --config_path client_config.json
```

2. Run the FL Client.

 ```bash
python minimal_fl_agent.py  --agent_name <AGENT_NAME> --num_rounds <NUM_ROUNDS>
```

## Further Reading

For detailed information on server-side components and integrating the STADLE `BasicClient`, refer to the [Usage section](https://stadle-documentation.readthedocs.io/en/latest/usage.html) in the STADLE documentation.
