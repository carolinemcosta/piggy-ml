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
  ''' Copies the data from the Dropbox folder to a local folder
      
      Parameters: 
        root_data (string): Root Dropbox directory with the point cloud data
        local_path (string): Local folder to copy the data to
        file_name (string): name of data file
        
      Returns:
        None
  '''  
  os.makedirs(local_path, exist_ok=True)
  local_data = os.path.join(local_path, file_name)
  copyfile(root_data,local_data) # make local copy
  
def load_pig_data(data_path=LOCAL_PATH, file_name=FILE_NAME):
  ''' Copies the data from the Dropbox folder to a local folder
      
      Parameters: 
        data_path (string): Local folder to copy the data to
        file_name (string): name of data file
        
      Returns:
        Pandas DataFrame with data read from the CSV file
  '''  
  csv_file = os.path.join(data_path, file_name)
  
  return pd.read_csv(csv_file) # return Pandas dataframe
  
