import yaml

def get_config(config_file):
    with open(config_file, "r") as file:
        config_dic = yaml.safe_load(file)
    return config_dic
