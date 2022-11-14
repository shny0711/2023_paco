import datetime
import yaml

def nowstr():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y.%m.%d %H:%M:%S')

def get_strdate():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_dict(path):
    with open(path, "r+") as f:
        data = yaml.unsafe_load(f)
    return data

def write_dict(path, data):
    with open(path, "w+") as f:
        f.write(yaml.dump(data, default_flow_style=False))