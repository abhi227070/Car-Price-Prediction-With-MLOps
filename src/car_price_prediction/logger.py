import logging
import os
from datetime import datetime

# Generate the log file name using the current date and time (e.g., "22_11_2024_14_45_12.log")
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Create the full path for the log file inside a 'logs' directory, in the current working directory
log_path = os.path.join(os.getcwd(), "logs")

# Ensure the 'logs' directory exists, if not, create it (exist_ok=True prevents error if directory already exists)
os.makedirs(log_path, exist_ok=True)

# Combine the directory path and log file name to get the full log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure the logging module to write log messages to the specified log file with a specific format
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file to write to
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format with timestamp, line number, logger name, log level, and message
    level=logging.INFO  # Set the logging level to INFO (logs messages of level INFO and above)
)
