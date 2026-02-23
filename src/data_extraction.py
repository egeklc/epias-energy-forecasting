import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import json

load_dotenv()


def get_tgt() -> str:
    """
    Takes EPİAŞ username and password.
    Returns TGT (Ticket Granting Ticket) to be used for EPİAŞ API calls.
    
    API Doc:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_adding_security_information_to_requests
    """
    login_url = "https://giris.epias.com.tr/cas/v1/tickets"
    
    headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/plain"
    }
    
    data = {
        "username": os.getenv("EPIAS_USERNAME"),
        "password": os.getenv("EPIAS_PASSWORD")
    }
    
    response = requests.post(
                login_url,
                data = data,
                headers = headers
    )
    status_code = response.status_code
    status_bool = response.ok
    
    if status_bool == True:
        
        if status_code != 201:
            print(f"TGT request status code: {status_code}")
            
        return response.text
        
    raise Exception(f"TGT request failed. Status code: {status_code}")


def generate_quarter_dates(year: int):
    """
    Takes year splits into quarters.
    Returns a list of tuple with start and end datetime in ISO-8601 format, suitable for API calls.
    """
    quarters = [(1,3), (4,6), (7,9), (10,12)]
    date_ranges = []

    for start_month, end_month in quarters:
        start_date = datetime(year, start_month, 1, 0, 0)
        if end_month == 12:
            end_date = datetime(year, 12, 31, 23, 0)
        else:
            end_date = datetime(year, end_month + 1, 1, 0, 0) - timedelta(hours=1)
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        date_ranges.append((start_str, end_str))
    return date_ranges

def get_year_datetime_range(year:int):
    """
    Takes year and returns start and end datetimes for a full year in ISO-8601 format .
    """
    start = datetime(year,1, 1, 0, 0, 0)
    end = datetime(year, 12, 31, 23, 0, 0)
    start_date = start.strftime("%Y-%m-%dT%H:%M:%S+03:00")
    end_date = end.strftime("%Y-%m-%dT%H:%M:%S+03:00")
    return start_date, end_date

def get_generation_data(tgt:str, start_date:str, end_date:str) -> pd.DataFrame:
    """
    Takes TGT (Ticket Granting Ticket), start date and end date.
    Maximum 3 months of data can be requested per API call.
    Returns generation data as a dataframe.
    
    API Doc:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_realtime-generation
    """
    
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/realtime-generation"
    headers = {
            "Accept-Language": "en",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "TGT": tgt
    }
    
    body = {
        "startDate": start_date,
        "endDate": end_date,
    }
    
    response = requests.post(
        url,
        json=body,
        headers=headers,
        timeout=300
    )
    status_code = response.status_code
    if status_code == 200:
        response_data = response.json()
        df = pd.DataFrame(response_data["items"])
        return df
    else:
        raise Exception(f"Request failed. Error code {status_code}: {response.text}")



def get_generation_data_yearly(tgt:str, year: int) -> pd.DataFrame:
    """
    Takes TGT (Ticket Granting Ticket) and year. Splits year into quarters.
    Each quarter is requested separately due to API limitations (max. 3 months per request).
    The results are concatenated into a single dataframe.
    
    API Doc:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_realtime-generation
    """
    date_ranges = generate_quarter_dates(year)
    quarter_df = []
    for start_date, end_date in date_ranges:
        df = get_generation_data(tgt=tgt, start_date=start_date, end_date=end_date)
        quarter_df.append(df)
    year_df = pd.concat(quarter_df, ignore_index=True)
    return year_df




def get_consumption_data(tgt:str, year: int) -> pd.DataFrame:
    """
    Takes TGT (Ticket Granting Ticket) and year.
    Returns hourly consumption data as a dataframe.
    API Doc:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_realtime-generation
    """
    
    start_date, end_date = get_year_datetime_range(year)
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"
    
    headers = {
            "Accept-Language": "en",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "TGT": tgt  
    }
    
    
    body = {
        "startDate": start_date,
        "endDate": end_date,
    }
    
    response = requests.post(
        url,
        json=body,
        headers=headers,
        timeout=300
    )
    status_code = response.status_code
    if status_code == 200:
        response_data = response.json()
        df = pd.DataFrame(response_data["items"])
        return df
    else:
        raise Exception(f"Request for year {year} failed. Error code {status_code}: {response.text}")