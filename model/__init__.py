from .nlp.BasicBert import BasicBert
from .nlp.BertPoint import BertPoint
from .nlp.BertPoolOut import BertPoolOut

model_list = {
    "BasicBert": BasicBert,
    "BertPoint": BertPoint,
    'BertPoolOut': BertPoolOut
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
