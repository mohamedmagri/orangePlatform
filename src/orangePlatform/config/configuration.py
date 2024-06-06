from orangePlatform.constants import *
from orangePlatform.utils.common import read_yaml ,create_directories
from orangePlatform.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig,
                                                 DataPreparationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig,
                                                 Model_TrainerConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        create_directories([config.root_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_preparation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ConvLSTM2D

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            num_lstm_units=params.num_lstm_units,
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            early_stopping_patience=params.early_stopping_patience,
            nsteps=params.nsteps
                    
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ConvLSTM2D

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            mlflow_uri="https://dagshub.com/mohamedmagri/orangePlatform.mlflow",
            num_lstm_units=params.num_lstm_units,
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            early_stopping_patience=params.early_stopping_patience,
            nsteps=params.nsteps
           
        )

        return model_evaluation_config
    

    def get_best_model_trainer_config(self) -> Model_TrainerConfig:
        config = self.config.best_model_trainer
        params = self.params.ConvLSTM2D

        create_directories([config.root_dir])

        model_trainer_config = Model_TrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            num_lstm_units=params.num_lstm_units,
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            early_stopping_patience=params.early_stopping_patience,
            nsteps=params.nsteps
                    
        )

        return model_trainer_config