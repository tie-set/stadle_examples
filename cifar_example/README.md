# CIFAR-10 Image Classification Example

These examples demonstrate an example of how STADLE can be used in order to integrate federated learning into existing PyTorch code for image classification on the CIFAR-10 dataset (`local_training.py`).

In order to run any of the STADLE examples, at least one aggregator must already be running.  In addition, the configuration file `client_config.json` must be modified to correspond with the server-side components if running on multiple instances.  If multiple (non-admin) agents are being created on the same machine, make sure to specify unique agent names using the `--agent_name` flag when running `fl_training.py` or the `agent_name` parameter in the `BasicClient` constructor.

Please refer to [Usage](https://stadle-documentation.readthedocs.io/en/latest/usage.html) for more details on starting the server-side components and the integration of the the STADLE client with the code located in `local_training.py`.

## Basic Client

The basic client example shows the barebones approach to integrate STADLE into the local training code.

### Execution

1. Upload the model. The base model information in the client config file is used to upload the model to the aggregator(s) for use in the FL process.

    ```bash
    stadle upload-model --config_path client_config.json
    ```

2. Run the Client

    ```bash
    python fl_training.py <options>
    ```

An imbalanced dataset can be created by specifying the specific classes to be weighted differently.  As an example, let's say we want to use all of the images for the classes we specify, and 25% of the images from the original CIFAR-10 dataset for the remaining classes.  In addition, let's specify the classes `airplane, automobile, bird` for this agent.  We would then run:

```bash
python fl_training.py --def_count 0.25 --sel_count 1.0 --classes airplane automobile bird
```