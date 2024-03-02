from stadle import AdminAgent
from stadle.lib.util import client_arg_parser
from stadle.lib.entity.model import BaseModel
from stadle import BaseModelConvFormat

from model import MyNet


def get_mynet_model():
    return BaseModel("PyTorch-Mnist-Model", MyNet(), BaseModelConvFormat.pytorch_format)


if __name__ == '__main__':
    args = client_arg_parser()

    admin_agent = AdminAgent(config_file="config/config_admin_agent.json", simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port,
                             model_path=args.model_path, base_model=get_mynet_model(),
                             agent_running=False)

    admin_agent.preload()
    admin_agent.initialize()
