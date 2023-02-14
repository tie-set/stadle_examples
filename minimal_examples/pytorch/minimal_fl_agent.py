import argparse
import random
from stadle import BasicClient

from minimal_model import MinimalModel

parser = argparse.ArgumentParser(description='STADLE Minimal Training')
parser.add_argument('--agent_name', default='pytorch_agent')
parser.add_argument('--num_rounds', type=int, default=20)
args = parser.parse_args()

client_config_file = 'client_config.json'
stadle_client = BasicClient(config_file=client_config_file, agent_name=args.agent_name)


model = MinimalModel()

for rnd in range(args.num_rounds):
    # Update the local model with the aggregate model weights
    state_dict = stadle_client.wait_for_sg_model().state_dict()
    model.load_state_dict(state_dict)

    # This is where the local training would occur
    # Performance metrics would be measured and passed back with the model to the aggregator
    perf_metrics = {
        'performance': random.random(),
        'accuracy': random.random(),
        'loss_training': random.random(),
        'loss_valid': random.random(),
        'loss_test': random.random()
    }

    # Send the locally trained model with the associated performance metric values
    stadle_client.send_trained_model(model, perf_values=perf_metrics)
    
    print(f'Sending model for round {rnd+1} to aggregator')

stadle_client.disconnect()
