import sys
import re
import torch.nn as nn
import models.residual_layer as residual_layer


elements = {
    'dropout': nn.Dropout,
    'batchnorm1d': nn.BatchNorm1d,
    'linear': nn.Linear,
    'relu': nn.ReLU,
    'residual': residual_layer.Residual
}


class Transform(nn.Module):
    """
        Transform document representation!
        transform_string: string for sequential pipeline
            eg relu#,dropout#p:0.1,residual#input_size:300-output_size:300-dropout:0.5
        params: dictionary like object for default params
            eg {emb_size:300}
    """

    def __init__(self, transform_string, params=None, device="cuda:0"):
        super(Transform, self).__init__()
        self.device = device
        modules = get_functions(transform_string, params)
        self.transform = nn.Sequential(*modules)

    def forward(self, embed):
        """
            Forward pass for transform layer
            Args:
                embed: torch.Tensor: document representation
            Returns:
                embed: torch.Tensor: transformed document representation
        """
        return self.transform(embed)

    def to(self):
        super().to(self.device)


def resolve_schema_args(string, ARGS):
    arguments = re.findall(r"@ARGS\.(.+?);", string)
    for arg in arguments:
        replace = '@ARGS.%s;' % (arg)
        to = str(ARGS[arg])
        if string.find('@ARGS.%s;' % (arg)) != -1:
            replace = '@ARGS.%s;' % (arg)
            if isinstance(ARGS[arg], str):
                to = str("\""+ARGS[arg]+"\"")
        string = string.replace(replace, to)
    return string


def get_functions(obj, params=None):
    obj_dict = []
    obj = resolve_schema_args(obj, params)
    for element in obj.split(','):
        key, params = element.split('#', 1)
        obj_dict.append([key, dict({})])
        if params != '':
            for param in params.split('-'):
                _key_param, _val_param = param.split(':', 1)
                obj_dict[-1][1][_key_param] = eval(_val_param)
    return list(map(lambda x: elements[x[0]](**x[1]), obj_dict))
