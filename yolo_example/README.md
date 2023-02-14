# DEPRECATED - YOLOv5 Example

### This example is currently deprecated, and is kept only for reference.

This prototype runs the previously tested YOLOv5 FL training process with STADLE on the vegetable dataset, using the new client API.

Currently, we capture the `mAP@[0.5,0.95]` (average of mean absolute precision values over [0.5,0.95] range of IoU thresholds) from the YOLOv5 val.py output to use as the 'test accuracy' of a model.

Two things must be done before execution.

1. The data should be located in the `yolo_example/dataset` folder (e.g. the eggplant data is in `yolo_example/dataset/data_eggplant`).
The data can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1qEMlg5Cz8YQJSMWG1m5H8_fCC9-vKTfy/view?usp=sharing).

2. Release 6.0 of YOLOv5 should be used; the source code can be downloaded from [this Github link](https://github.com/ultralytics/yolov5/archive/refs/tags/v6.0.zip).
The `yolov5` folder should be extracted into the `yolo_example` folder (i.e. the `yolov5` folder should be located at `yolo_example/yolov5`).
OR
Execute following bash commands, in order, from this folder[prototypes/examples/yolo_example/] as base.

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5/
    git reset --hard 956be8e642b5c10af4a1533e09084ca32ff4f21f
    ```

## Execution on localhost / single machine setup

Run the following commands in order to start the YOLOv5 FL example on the vegetable dataset.

The persistence server and the aggregator server should both be run from the root `stadle_dev` folder, and the admin agent/yolo engine should be run from the prototype folder (`prototypes/examples/yolo_example`).

1. Upload the model. The admin agent is used to upload the model to the aggregator(s) and database for use in the FL process.

    ```bash
    python yolo_admin_agent.py
    ```

    At this stage, we are ready to run the agents on each of the vegetable datasets - the dataset used by each agent can be specified through the respective CLI argument.

2. Run the client on the specified dataset

    ```bash
    python yolo_engine_sim.py --dataset <eggplant/negi/tomato>
    ```

    Accepted arguments:

    - `--dataset`: One of `eggplant/negi/tomato`; specifies which dataset the agent should use for local training
    - `--start_round`: Integer >= 0; specifies the round at which the agent should start training and sending models.  Defaults to 0.
    - `--termination_round`: Integer >= 0; specifies the round at which the agent should stop training and sending models.  If not provided, the agent will run indefinitely.
    - `--device`: Either `cpu`, or an integer/list of integers corresponding to the device numbers of the GPU(s) to be used; specifies device(s) to run YOLO with.  If not provided, will select based on local PyTorch installation.
