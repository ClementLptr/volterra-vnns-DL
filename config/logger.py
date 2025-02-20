import logging
import os
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime

def load_config(config_file='config.json'):
    """Load configuration from the given JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), config_file)
    
    with open(config_path, 'r') as file:
        return json.load(file)

def setup_logger(config_file='config.json'):
    """Set up the logger with configurations from config file."""
    # Load configuration
    config = load_config(config_file)
    logger_config = config['logger']
    model_config = config['model']
    task_config = config['task']

    # Extract logger settings from the configuration
    log_dir = logger_config.get('log_dir', 'logs')
    log_level = logger_config.get('log_level', 'INFO')
    max_log_size = logger_config.get('max_log_size', 10485760)
    backup_count = logger_config.get('backup_count', 5)
    model_version = logger_config.get('model_version', 'Not specified')
    model_name = model_config.get('model_name', 'UnknownModel')
    task_type = task_config.get('task_type', 'train')

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_dir, model_version)
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"{model_name}-{model_version}-{task_type}-run-{timestamp}.log"
    log_path = os.path.join(log_dir, log_file)

    # Set up the logger
    logger = logging.getLogger('VideoClassification')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        # Console handler for displaying logs on the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Rotating file handler for saving logs to a file
        rotating_file_handler = RotatingFileHandler(log_path, maxBytes=max_log_size, backupCount=backup_count)
        rotating_file_handler.setLevel(logging.DEBUG)

        # Formatter for logging messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        rotating_file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(rotating_file_handler)

        # Log initialization messages
        logger.info(f'Logging initialized. Log file: {log_path}')
        logger.info(f'Model version: {model_version}')
    
    return logger
