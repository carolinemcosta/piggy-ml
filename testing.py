import os
import pandas as pd
import get_data as gd

def main():
  # fetch data from Dropbox
  gd.fetch_pig_data()
  
  # load data to dataframe
  pig_data = gd.load_pig_data()
  print(pig_data.head())
  
if __name__== "__main__":
  main()
