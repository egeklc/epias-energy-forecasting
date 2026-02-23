import os
import time
from src.data_extraction import get_tgt, get_consumption_data, get_generation_data_yearly

save_dir = "data/raw"
year_list = range(2018,2026)

def extract_raw_data():
    os.makedirs(save_dir, exist_ok=True)
    
    tgt = get_tgt()
    
    for year in year_list:
        file_path_consumption = os.path.join(save_dir, f"consumption_data_{year}.csv")
        file_path_generation = os.path.join(save_dir, f"generation_data_{year}.csv")
        
        print(f"Extracting consumption data for year {year}")
        df_consumption = get_consumption_data(tgt=tgt, year=year)
        df_consumption.to_csv(file_path_consumption, index=False)
        print(f"Successfully saved consumption data for year {year} to {file_path_consumption}")
        
        time.sleep(2)
        
        print(f"Extracting generation data for year {year}")
        df_generation = get_generation_data_yearly(tgt=tgt, year=year)
        df_generation.to_csv(file_path_generation, index=False)
        print(f"Successfully saved generation data for year {year} to {file_path_generation}")
        
        time.sleep(2)

if __name__ == "__main__":
    extract_raw_data()