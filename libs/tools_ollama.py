'''
Helper function for ollama
launch : 
- check if model exists, if not, download it, 
- prepare it (change parameters)
'''
import ollama
from libs.utilities import getconfig

CONFIG = "main"
MDL = getconfig(CONFIG)["mainmodel"]  # mainmodel=llama3
MDL_NAME = getconfig(CONFIG)["mainmodel_name"]
TEMPERATURE = getconfig(CONFIG)["temperature"]
TOP_P = getconfig(CONFIG)["top_p"]


def launch_server_ollama(mdl=MDL, temperature=TEMPERATURE, top_p=TOP_P, name=MDL_NAME, config=None):
    '''
    Launch ollama model
    '''
    # get config param
    if config is not None:
        mdl, name, temperature, top_p = get_ollama_conf(config)
    # download model
    ollama.pull(mdl)
    # prepare model initialization
    modelfile = f'''
    FROM {mdl}
    PARAMETER temperature {temperature}
    PARAMETER top_p {top_p}
    '''
    ollama.create(model=name, modelfile=modelfile)
    return name


def get_ollama_conf(config=CONFIG):
    '''
    Get ollama config
    '''
    mdl = getconfig(config)["mainmodel"]
    name = getconfig(config)["mainmodel_name"]
    temperature = getconfig(config)["temperature"]
    top_p = getconfig(config)["top_p"]

    print(f"Temperature : {temperature} Top_p : {top_p}")

    return mdl, name, temperature, top_p
