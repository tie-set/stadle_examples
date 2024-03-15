# STADLE Examples

<img src="stadle_logo.png" width="600"/>

The STADLE (Scalable, Traceable, Adaptive, Distributed Learning) platform is a paradigm-shifting technology for collaborative and continuous learning combining privacy-preserving frameworks.
This repo contains some demo examples on how STADLE can be easily integrated into existing ML code.

The public documentation for STADLE can be found [here](https://stadle-documentation.readthedocs.io/en/latest/usage.html).

## Prerequisites

Before running the STADLE examples, ensure the following conditions are met:

- At least one aggregator is running on the [STADLE Dashboard](https://www.stadle.ai/).
- You have a valid license key for STADLE. If you need one, sign up at TieSet's [User Portal](https://userportal.tieset.ai/) and purchase one or more STADLE licenses.
- The license key is to be set in the User Profile page of the STADLE dashboard.
- Create a project with a valid license key, and then initiate an aggregator.
- Check the `Address to Connect` displayed on the STADLE Dashboard page.

## PyPI Installation

Befire executing the example code, please install `stadle-client` library from the PyPI server as follows:

```bash
pip install --upgrade pip
pip install stadle-client
```

If the installation above does not work, try installing `stadle-client` with the follwoing command:
```bash
pip install --no-deps stadle-client
```
After that manually install the required dependencies.
```bash
pip install certifi==2021.10.8
pip install click==8.0.3
pip install getmac==0.8.2
# Continue with other dependencies as needed
```


## Tech Support

If you have any issues, please reach out to our technical support team via [support@tie-set.com](support@tie-set.com).


