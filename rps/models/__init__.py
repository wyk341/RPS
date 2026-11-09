import copy
from .convertor import TableMasterConvertor
from .model import RPSModel


def build_convertor(config):
    values=copy.deepcopy(dict(config)); values.pop('type',None)
    return TableMasterConvertor(**values)


def build_model(config):
    values=copy.deepcopy(dict(config)); values.pop('type',None)
    return RPSModel(**values)
