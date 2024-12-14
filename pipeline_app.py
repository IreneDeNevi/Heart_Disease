import sys
import time
import itertools
import argparse
import threading

from automl import AutoML
import util
import importlib
importlib.reload(util)

# Logger configuration
logger = util.get_logger('PipelineApp', log_file='application.log')


def show_loading_message(message="Processing, please do not close this window"):
  """
  Display a loading message with a rotating spinner in the console.
  
  Args:
    message (str): The message to display.
  """
  spinner = itertools.cycle(['-', '\\', '|', '/'])  # cycle for the wheel
  sys.stdout.write(f"\n{message}... ")  # Initial message
  sys.stdout.flush()
  
  try:
    while True:
      sys.stdout.write(next(spinner))  # Write next symbol
      sys.stdout.flush()
      time.sleep(0.3)  # Pause for animation's speed
      sys.stdout.write('\b')  # Back to previous character
  except KeyboardInterrupt:
    sys.stdout.write("\nOperation interrupted by user.\n")
    sys.stdout.flush()


# Main Function 
if __name__ == "__main__":
  def run_loading():
    show_loading_message()

  parser = argparse.ArgumentParser(description="Run a machine learning pipeline for heart disease data")
  parser.add_argument("csv_path", help="Path to the cleaned CSV file")
  parser.add_argument("target_column", help="Target column name")
  parser.add_argument("algorithm", help="ML algorithm to use")
  parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data")
  parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
  parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
  parser.add_argument("--model_name", type=str, default="model", help="Name of the resulting .pkl file")
  args = parser.parse_args()

  loading_thread = threading.Thread(target=run_loading)
  loading_thread.daemon = True
  loading_thread.start()

  try:

    util.check_mandatory_parameters(args.csv_path, args.target_column, args.algorithm)

    automl = AutoML()
    automl.init(args.csv_path, args.target_column, args.algorithm, args.random_state, args.model_name)
    results = automl.train_and_evaluate(args.test_size, args.random_state, args.cv)
    
    print("\nProcess completed successfully! Check the log file 'application.log' for details.")

  except Exception as e:
    print("\nCritical error! Check the log file 'application.log' for details.")
    logger.critical("Critical error in main: %s", str(e))

