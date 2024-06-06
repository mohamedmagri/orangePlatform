from orangePlatform.config.configuration import ConfigurationManager
from orangePlatform.components.best_model_training import BestModelTrainer
from orangePlatform import logger




STAGE_NAME = "Model hyperparameter tuning stage"

class BestModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_best_model_trainer_config()
        model_trainer_config = BestModelTrainer(config=model_trainer_config)
        model_trainer_config.train()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BestModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

