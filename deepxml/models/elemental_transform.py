import sys
import re
import torch.nn as nn
import models.residual_layer as residual_layer
import json

"""
example usage 
python elemental_transform.py "relu#,dropout#p:0.1,residual#input_size:300-output_size:300-dropout:0.5"
python elemental_transform.py "relu#,dropout#p:0.1,residual#input_size:@ARGS.emb_size;-output_size:@ARGS.emb_size;-dropout:0.5"
"""

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

    def __init__(self, modules, device="cuda:0"):
        super(Transform, self).__init__()
        self.device = device
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

def resolve_schema_args(jfile, ARGS):
	arguments = re.findall(r"#ARGS\.(.+?);", jfile)
	for arg in arguments:
		replace = '#ARGS.%s;' % (arg)
		to = str(ARGS.__dict__[arg])
		if jfile.find('\"#ARGS.%s;\"' % (arg)) != -1:
			replace = '\"#ARGS.%s;\"' % (arg)
			if isinstance(ARGS.__dict__[arg],str):
				to = str("\""+ARGS.__dict__[arg]+"\"")
		jfile = jfile.replace(replace,to)
	return jfile

def fetch_json(file, ARGS):
	with open(file, encoding='utf-8') as f:
		file = ''.join(f.readlines())
		schema = resolve_schema_args(file, ARGS)
	return json.loads(schema)

def get_functions(obj, params=None):
    return list(map(lambda x: elements[x](**obj[x]), obj['order']))


if __name__ == "__main__":
    obj = sys.argv[1]
    obj = fetch_json(obj, params)['transform_fine']
    list_of_functions = Transform(get_functions(obj), None)
    print(list_of_functions)
