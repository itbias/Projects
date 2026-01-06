import yaml

def upload_config(data: dict, path: str) -> None:
    with open(path, "w") as file:
        yaml.dump(data, file) 