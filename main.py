import config
from SQUAD.prepare_data import download_squad, prepare_csv


if __name__ == "__main__":
    ## Downloading squad train and dev dataset
    # download_squad(config.SQUAL_URL, config.SQUAD_TRAIN_FILE_NAME, config.SQUAD_OUTPUT_DIR)
    # download_squad(config.SQUAL_URL, config.SQUAD_DEV_FILE_NAME, config.SQUAD_OUTPUT_DIR)
    prepare_csv()
    
    