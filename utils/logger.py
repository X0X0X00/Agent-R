import logging
import os
import datetime

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
# disable stdout
# Remove all handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
PROJECT_WORKDIR_PATH = os.getenv("PROJECT_WORKDIR_PATH", ".")
log_dir = os.path.join(PROJECT_WORKDIR_PATH, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Generate filename with current datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f"app-{current_time}.log"
path = os.path.join(log_dir, log_filename)
 
fn = logging.FileHandler(path)
fn.setFormatter(formatter)
logger.addHandler(fn)
logger.setLevel(logging.DEBUG)


# import logging
# import os
# import datetime

# # Configure root logger
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")

# # Create a global logger with a specific name (not __name__)
# logger = logging.getLogger("mcts_logger")  # Use a constant name

# # Setup file logging
# PROJECT_WORKDIR_PATH = os.getenv("PROJECT_WORKDIR_PATH", ".")
# log_dir = os.path.join(PROJECT_WORKDIR_PATH, 'logs')
# os.makedirs(log_dir, exist_ok=True)

# # Generate filename with current datetime
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_filename = f"app-{current_time}.log"
# path = os.path.join(log_dir, log_filename)

# # Add file handler
# file_handler = logging.FileHandler(path)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# # Add console handler to see logs in terminal
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# logger.setLevel(logging.DEBUG)

# # Don't propagate to root logger to avoid duplicate logs
# logger.propagate = False