import logging
# def get_logger(name='default_logger'):
#   """
#   Configure and return a shared logger

#   Args:
#     name (str): The name of the logger.
  
#   Returns:
#     The configured logger
#   """
#   logger = logging.getLogger(name)
#   if not logger.hasHandlers():  # Evita di aggiungere più handler
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
#     file_handler = logging.FileHandler('application.log', mode='w')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     logger.setLevel(logging.DEBUG)
#   return logger

def get_logger(name, log_file=None, level=logging.INFO):

  """
  Configure and return a shared logger

  Args:
    name (str): The name of the logger.
  
  Returns:
    The configured logger
  """
  logger = logging.getLogger(name)
  logger.setLevel(level)
  
  # Evita di aggiungere handler multipli
  if not logger.hasHandlers():
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    
    if log_file:
      file_handler = logging.FileHandler(log_file, mode='w')
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)
  
  return logger

def is_none_or_empty(input_string: str):
  """
  Checks if a string is None or empty.

  Args:
    input_string (str): The string to check.

  Returns:
    bool: True if the string is None or empty, otherwise False.
  """
  return input_string is None or input_string.strip() == ""

def check_mandatory_parameters(csv_path: str, target_column: str, algorithm: str):
  """
  Checks if any of the three string parameters is None or empty.

  Args:
      param1 (str): The first parameter to check.
      param2 (str): The second parameter to check.
      param3 (str): The third parameter to check.

  Raises:
      ValueError: If any parameter is None or empty.
  """
  if is_none_or_empty(csv_path):
    raise ValueError("The csv_path parameter is mandatory and cannot be None or empty.")
  if is_none_or_empty(target_column):
    raise ValueError("The target_column parameter is mandatory and cannot be None or empty.")
  if is_none_or_empty(algorithm):
    raise ValueError("The algorithm parameter is mandatory and cannot be None or empty.")

