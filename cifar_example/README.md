# CIFAR-10 Image Classification Example

These examples demonstrate two different ways that STADLE can be used in order to integrate federated learning into existing PyTorch code for image classification on the CIFAR-10 dataset (`local_training.py`).

In order to run any of the STADLE examples, at least one aggregator and one persistence-server must already be running.  In addition, the configuration files located in the `config` directory must be modified to correspond with the server-side components if running on multiple instances.  If multiple (non-admin) agents are being created on the same machine, make sure to specify unique agent names using the `--agent_name` flag when running `cifar_fl_agent.py` or the `agent_name` parameter in the `BasicClient` constructor.

Please refer to [Usage](https://stadle-documentation.readthedocs.io/en/latest/usage.html) for more details on starting the server-side components and the integration of the two STADLE client types with the code located in `local_training.py`.

## Basic Client

The basic client example shows the barebones approach to integrate STADLE into the local training code.

### Execution

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

    ```bash
    python cifar_admin_agent.py
    ```

2. Run the Client

    ```bash
    python fl_training.py
    ```

## Integrated Client

The integrated client example demonstrates an alternate approach for integrating local training code into a STADLE agent.  This example also allows for imbalaced versions of the CIFAR-10 dataset to be constructed and used during local training, allowing for testing of the robustness from FL.

### Execution

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

    ```bash
    python cifar_admin_agent.py
    ```

2. Run the Client

    ```bash
    python cifar_engine_sim.py
    ```

An imbalanced dataset can be created by specifying the specific classes to be weighted differently.  As an example, let's say we want to use all of the images for the classes we specify, and 25% of the images from the original CIFAR-10 dataset for the remaining classes.  In addition, let's specify the classes `airplane, automobile, bird` for this agent.  We would then run:

```bash
python cifar_engine_sim.py --def_count 0.25 --sel_count 1.0 --airplane --automobile --bird
```

The number of local training epochs can be set with the argument `--lt_epochs <num_epochs>`, and the learning rate can be set with the argument `--lr <learning_rate>`.

The model checkpoint state dicts can be found in the `checkpoint` folder.
