from orangePlatform.config.configuration import ConfigurationManager
from orangePlatform.components.data_preparation import DataPreparation
from orangePlatform import logger
from pathlib import Path

STAGE_NAME = "Data Preparation stage"

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_preparation_config = config.get_data_preparation_config()
                data_preparation = DataPreparation(config=data_preparation_config)
                data_preparation.train_test_spliting()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)





if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
