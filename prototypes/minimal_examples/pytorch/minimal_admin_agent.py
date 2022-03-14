from stadle import AdminAgent
from stadle.lib.util import admin_arg_parser
from .minimal_model import MinimalModel
from stadle.lib.entity.model import BaseModel
from stadle import BaseModelConvFormat


def get_minimal_model():
    return BaseModel("PyTorch-Minimal-Model", MinimalModel(), BaseModelConvFormat.pytorch_format)


if __name__ == '__main__':
    args = admin_arg_parser()

    admin_agent = AdminAgent(config_file=args.config_path, simulation_flag=args.simulation,
                             aggregator_ip_address=args.ip_address, reg_socket=args.reg_port,
                             exch_socket=args.exch_port, model_path=args.model_path, base_model=get_minimal_model(),
                             agent_running=args.agent_running)

    admin_agent.preload()
    admin_agent.initialize()
