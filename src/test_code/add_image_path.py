import pandas as pd
import pickle as pkl
import os
import re
import fnmatch
from tqdm import tqdm

def find_image_path(id,origin_path):
    """
        find image path
    """
    files_list = os.listdir(origin_path)
    pattern = f"{id}.*"
    matching_files = fnmatch.filter(files_list,pattern)
    
    if matching_files:
        image_path = os.path.join(origin_path,matching_files[0])
        return image_path
    else:
        return None
if __name__ == "__main__":
    with open("/data1/zzy/FakeNewsCode/fake-news-baselines/data/SIGIR24/Snopes/add_label_snopes_3.pkl","rb") as f:
        data_df = pkl.load(f)
    print(data_df.columns)
    print(len(data_df))
    print(len(data_df)*0.6)
    print(len(data_df)*0.2)
    