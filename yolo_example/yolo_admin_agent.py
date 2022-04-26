from stadle import AdminAgent
from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel
from stadle.lib.util import client_arg_parser

from yolov5.models.yolo import Model

import torch

if __name__ == '__main__':
    args = client_arg_parser()

    # Create model object
    weight_file = './dataset/weight/eggplant_1000.pt'

    base_model_obj = torch.load(weight_file)

    #base_model_obj = train_engine_yolo(data,save_dir,batch_size=16,epochs=1,weights=weight_file,cfg=data,project_dir='./data/yolo_demo/base_model')

    base_model = BaseModel('YOLOv5 Vegetable Model', base_model_obj['model'], BaseModelConvFormat.pytorch_format)

    admin_agent = AdminAgent(config_file=args.config_path, simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port,
                             exch_port=args.exch_port, model_path=args.model_path, base_model=base_model,
                             agent_running=args.agent_running)

    admin_agent.preload()
    admin_agent.initialize()
