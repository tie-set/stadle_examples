from stadle import AdminAgent
from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel
from stadle.lib.util import admin_arg_parser

from vgg import VGG as Model


def get_base_model():
    return BaseModel("PyTorch-CIFAR10-Model", Model('VGG16'), BaseModelConvFormat.pytorch_format)


if __name__ == '__main__':
    args = admin_arg_parser()

    admin_agent = AdminAgent(config_file=args.config_path, simulation_flag=args.simulation,
                             aggregator_ip_address=args.ip_address, reg_socket=args.reg_port,
                             exch_socket=args.exch_port, model_path=args.model_path, base_model=get_base_model(),
                             agent_running=args.agent_running)

    admin_agent.preload()
    admin_agent.initialize()
