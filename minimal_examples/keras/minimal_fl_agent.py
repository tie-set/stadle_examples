import argparse
import random
from tensorflow import keras
from stadle import BasicClient

# from minimal_model import MinimalModel

def get_minimal_model():
    model = keras.models.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=(3,)),
        keras.layers.Dense(4)
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='STADLE Minimal Training')
    parser.add_argument('--agent_name', default='keras_agent')
    parser.add_argument('--num_rounds', type=int, default=20)
    args = parser.parse_args()

    client_config_file = 'client_config.json'
    stadle_client = BasicClient(config_file=client_config_file, agent_name=args.agent_name)


    model = get_minimal_model()

    for rnd in range(args.num_rounds):
        # Update the local model with the aggregate model weights
        model = stadle_client.wait_for_sg_model()

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