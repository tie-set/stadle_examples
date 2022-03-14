# STADLE Examples

<img src="logo/stadle_logo.png" width="600"/>

Our STADLE platform is a paradigm-shifting technology for collaborative and continuous learning combining privacy-preserving frameworks.
STADLE platform stands for Scalable, Traceable, Adaptive, Distributed Learning platform for versatile ML applications.
In this repo, we will procide several examples into which STADLE's client-side libraries are integrated.

## Table of Contents

- [STADLE v2.3.0](#stadle-v20)
  - [Table of Contents](#table-of-contents)
  - [General Terminologies](#general-terminologies)
  - [Usage](#usage)
  - [Tech Support](#tech-support)

## General Terminologies

There are 3 main components in STADLE.

- Persistence-server

  - A core functionality which helps in keeping track of various database entries.
  - Packaged as a command inside `stadle` library.
  - `stadle persistence-server [args]`

- Aggregator

  - A core functionality which helps aggregation process.
  - Packaged as a command inside `stadle` library.
    - `stadle aggregator [args]`

- Client
  - In charge of communicating with `stadle` core functions.
  - A core functionality which helps executing the machine learning code from client side.
  - Packaged inside `stadle` library as a class.
    - `from stadle import IntegratedClient`
    - `class IntegratedClient` is used to let `stadle` know that the following code is going to be ML.

## Usage

This repository hosts only the examples/prototypes for using stadle.
Currently, cifar_example is the only end-to-end working example.
New prototypes will be added as we proceed forward.

To execute any example, one needs to know TieSet provided `aggregator-IP-address` or should have stadle server code to execute various examples.
The code provided here is a very basic level code to get user familiar with structure.

### Execute cifar_example

    - Install stadle_client package.
        - `pip install --index-url http://[pypiserver_ip]:[port_number] stadle_client --trusted-host [pypiserver_ip] --extra-index-url https://pypi.org/simple`
        - `pypiserver_ip` and `port_number` can be acquired by contacting our sales team.
    - Setup config files to use correct `aggregator-IP-address`.
        - This is denoted in the `config/config_admin_agent.json` file as `aggr_ip`.
        - If the servers are run locally, this will be `localhost`.
        - If Tieset provided servers are used, set the value appropriately.
    - Execute admin_agent code.
        - Admin agent sets up the model and other required details for later execution.
        - One can run `python cifar_admin_agent.py` to run provided model through the process.
    - Execute federated_learning agent.
        - This will do the actual model training on the local machine and upload performance matrices to stadle servers and in the meantime, receieve the agumented models.
        - One can run `python cifar_fl_agent.py`.


## Tech Support

If you have any issues, please reach out to our technical support team via [support@tie-set.com](support@tie-set.com).


