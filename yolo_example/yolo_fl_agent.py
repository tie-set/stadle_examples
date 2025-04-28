import os
import argparse

import torch

from ultralytics import YOLO

from stadle import BasicClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='STADLE Minimal Training')
    parser.add_argument('--agent_name')
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--dataset', type=str, help='Choose local dataset to train on (eggplant/negi/tomato)')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    client_config_file = 'client_config.json'

    agent_name = (f'{args.dataset}_yolo_agent' if args.agent_name is None else args.agent_name)

    stadle_client = BasicClient(config_file=client_config_file, agent_name=args.agent_name)

    data_yaml = os.path.join(os.getcwd(), f'dataset/data_{args.dataset}/{args.dataset}_data.yaml')

    device = args.device

    # YOLO object contains model alongside other training/validation related objects
    # We use the internal model object with STADLE
    model_container = YOLO("yolo11n.yaml")
    # Model architecture changes when training begins so we use post-training checkpoint to get true architecture
    model = torch.load('base_yolo_model.pt').to(device)

    num_epochs = 25

    for rnd in range(args.num_rounds):
        # Update the local model with the aggregate model weights
        smodel, model_id = stadle_client.wait_for_sg_model(get_sg_model_id=True)

        # Load state dict from current semi-global model and store in YOLO internal model
        model.load_state_dict(smodel.state_dict())
        model_container.model = model

        # Get semi-global model validation results
        sg_val_results = model_container.val(
            data=data_yaml,
            device=device
        )

        train_results = model_container.train(
            data=data_yaml,
            epochs=num_epochs,
            warmup_epochs=0,
            device=device
        )

        # Get local model validation results
        val_results = model_container.val(
            data=data_yaml,
            device=device
        )

        perf_metrics = {}
        sg_perf_metrics = {}

        # Extract training/validation metrics from YOLO result dicts for local model
        for k in val_results.results_dict.keys():
            trim_k = k.split('/')[1][:-3] if len(k.split('/')) > 1 else k
            perf_metrics['val_' + trim_k] = val_results.results_dict[k]
            perf_metrics['train_' + trim_k] = train_results.results_dict[k]

        # Extract training/validation metrics from YOLO result dicts for semi-global model
        for k in sg_val_results.results_dict.keys():
            trim_k = k.split('/')[1][:-3] if len(k.split('/')) > 1 else k
            sg_perf_metrics['val_' + trim_k] = sg_val_results.results_dict[k]

        # Send the locally trained model with the associated performance metric values
        stadle_client.send_trained_model(model_container.model, perf_values=perf_metrics)

        # Send the semi-global model validation metrics
        stadle_client.send_metrics(model_id, sg_perf_metrics)
        
        print(f'Sending model for round {rnd+1} to aggregator')

    stadle_client.disconnect()
