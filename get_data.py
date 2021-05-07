import os
import pandas as pd
from shutil import copyfile
from pathlib import Path

# the data directory is hardcoded for now until we are able to make the data public
ROOT_DIR = "%s/Dropbox/PigEPdata/point-cloud-data/"%(Path.home())
FILE_NAME = "pigs-for-ml.csv"
ROOT_DATA = ROOT_DIR + FILE_NAME
LOCAL_PATH = os.path.join("datasets", "pigs")

def fetch_pig_data(root_data=ROOT_DATA, local_path=LOCAL_PATH, file_name=FILE_NAME):
  os.makedirs(local_path, exist_ok=True)
  local_data = os.path.join(local_path, file_name)
  copyfile(root_data,local_data) # make local copy
  
def load_pig_data(data_path=LOCAL_PATH, file_name=FILE_NAME):
  csv_file = os.path.join(data_path, file_name)
  return pd.read_csv(csv_file) # return Pandas dataframe
  
