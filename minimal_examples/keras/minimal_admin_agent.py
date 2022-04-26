import tensorflow as tf
from tensorflow import keras

from stadle import AdminAgent
from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel
from stadle.lib.util import client_arg_parser


def get_minimal_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=(3,)),
        keras.layers.Dense(4)
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return BaseModel("Tensorflow-Minimal-Model", model, BaseModelConvFormat.keras_format)


if __name__ == '__main__':
    args = client_arg_parser()

    admin_agent = AdminAgent(config_file=args.config_path, simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port,
                             exch_port=args.exch_port, model_path=args.model_path, base_model=get_minimal_model(),
                             agent_running=args.agent_running)

    admin_agent.preload()
    admin_agent.initialize()
