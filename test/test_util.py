import pytest
import logging
import os
import sys
# include the project root directory (Heart_Disease) in the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import get_logger, is_none_or_empty, check_mandatory_parameters

def test_get_logger_creates_logger_with_file():
  logger = get_logger("test_logger", "test.log", level=logging.DEBUG)
  assert isinstance(logger, logging.Logger)
  assert logger.hasHandlers()

def test_get_logger_without_file():
  logger = get_logger("test_logger_no_file", level=logging.INFO)
  assert isinstance(logger, logging.Logger)
  assert logger.hasHandlers()

def test_is_none_or_empty_with_none():
  assert is_none_or_empty(None) == True

def test_is_none_or_empty_with_empty_string():
  assert is_none_or_empty("") == True

def test_is_none_or_empty_with_whitespace():
  assert is_none_or_empty("   ") == True

def test_is_none_or_empty_with_valid_string():
  assert is_none_or_empty("valid") == False

def test_check_mandatory_parameters_valid_inputs():
  try:
    check_mandatory_parameters("file.csv", "target_column", "algorithm")
  except ValueError:
    pytest.fail("check_mandatory_parameters raised ValueError unexpectedly!")

def test_check_mandatory_parameters_with_empty_csv_path():
  with pytest.raises(ValueError, match="The csv_path parameter is mandatory and cannot be None or empty."):
      check_mandatory_parameters("", "target_column", "algorithm")

def test_check_mandatory_parameters_with_empty_target_column():
  with pytest.raises(ValueError, match="The target_column parameter is mandatory and cannot be None or empty."):
    check_mandatory_parameters("file.csv", "", "algorithm")

def test_check_mandatory_parameters_with_empty_algorithm():
  with pytest.raises(ValueError, match="The algorithm parameter is mandatory and cannot be None or empty."):
    check_mandatory_parameters("file.csv", "target_column", "")
