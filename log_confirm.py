import json
import yaml

if __name__ == "__main__":
    with open("./config/config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"./wandb/{CFG['wandb_log_folder']}/files/wandb-summary.json", encoding="utf-8") as f:
        log_data = json.load(f)

    print(log_data)