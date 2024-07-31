import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import utm
import numpy as np
import re
from utils import from_mrk_to_csv,get_rgb_data


def get_geoid_height_egm2008(latitude, longitude):
    """
    Gets the geoid height from the EGM2008 model for a given latitude and longitude.

    Parameters:
    latitude (float): The latitude of the point.
    longitude (float): The longitude of the point.

    Returns:
    float or None: The geoid height if successful, otherwise None.
    """
    try:
        url = f"https://geographiclib.sourceforge.io/cgi-bin/GeoidEval?input={latitude}%20{longitude}&model=EGM2008"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pre_tag = soup.find('pre')
            if pre_tag:
                lines = pre_tag.get_text().strip().split('\n')
                for line in lines:
                    if "EGM2008" in line:
                        geoid_height = float(line.split('=')[1].strip())
                        return geoid_height
            raise Exception("Unable to find geoid height value in HTML response")
        else:
            raise Exception(f"Request error: {response.status_code}")
    except Exception as e:
        print(f"Error getting geoid height for lat={latitude}, lon={longitude}: {e}")
        return None


def calculate_orthometric_height(row):
    """
    Calculates the orthometric height for a given DataFrame row.

    Parameters:
    row (pd.Series): The row of the DataFrame.

    Returns:
    float or None: The orthometric height if successful, otherwise None.
    """
    latitude = row['lat']
    longitude = row['lon']
    ellh = row['Ellh']
    
    geoid_height = get_geoid_height_egm2008(latitude, longitude)
    if geoid_height is not None:
        orthometric_height = ellh - geoid_height
        return orthometric_height
    else:
        return None


def fromGPStoUTM(lon, lat):
    """
    Converts GPS coordinates to UTM coordinates.

    Parameters:
    lon (pd.Series): The longitude series.
    lat (pd.Series): The latitude series.

    Returns:
    tuple: Two numpy arrays containing the UTM coordinates.
    """
    Y, X = [], []
    for i in np.arange(len(lat)):
        LAT, LON, _, _ = utm.from_latlon(lat[i], lon[i])
        X = np.append(X, LAT)
        Y = np.append(Y, LON)
    return X, Y


def process_photos(df, path_img_dir, result):
    """
    Processes photos to extract metadata and create a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with the GPS data.
    result (tuple): The UTM coordinates.

    Returns:
    pd.DataFrame: The processed DataFrame with photo metadata.
    """
    lista = []
    for filename in os.listdir(path_img_dir):
        if filename.lower().endswith('_d.jpg'):
            source_file = os.path.join(path_img_dir, filename)
            match = re.search(r'_\d{14}_(\d{4})_', filename)
            if match:
                valore_estratto = int(match.group(1))
                rgb_data=get_rgb_data(source_file)
                GimbalYawDegree = rgb_data['GimbalYawDegree']
                GimbalRollDegree = rgb_data['GimbalRollDegree']
                RelativeAltitude = rgb_data['RelativeAltitude']
                YawDegree = GimbalYawDegree
                riga = {
                    "id_photo": filename,
                    "E": result[0][valore_estratto - 1],
                    "N": result[1][valore_estratto - 1],
                    "H": df['H'][valore_estratto - 1],
                    "YawDegree": YawDegree,
                    "GimbalRollDegree": GimbalRollDegree,
                    "RelativeAltitude": RelativeAltitude
                }
                lista.append(riga)
    return pd.DataFrame(lista)


if __name__ == "__main__":

    # Path to the uploaded CSV file
    path_file_mrk = './raw_data/DJI_202405251220_005_cimadimelfimisto_Timestamp.MRK'
    csv_file_path = './resources/timestamp/005_cimadimelfi_Timestamp.csv'
    updated_csv_file_path = './resources/timestamp/005_cimadimelfi_Timestamp_extended.csv'
    path_navinfo_csv = './resources/navinfo/005_cimadimelfi_navinfo.csv'
    path_img_dir = './raw_data/DJI_202405251220_005_cimadimelfimisto'

    # Load the CSV file into a DataFrame
    from_mrk_to_csv(path_file_mrk, csv_file_path)
    df = pd.read_csv(csv_file_path)

    # Apply the function to each row to calculate the new 'H' column
    df['H'] = df.apply(calculate_orthometric_height, axis=1)

    # Remove rows with None values in 'H'
    df = df.dropna(subset=['H'])

    # Save the updated DataFrame as a new CSV file
    df.to_csv(updated_csv_file_path, index=False)
    print(f"Updated CSV file saved to: {updated_csv_file_path}")

    # Load the updated CSV file and process photos
    df = pd.read_csv(updated_csv_file_path)
    lat = df['lat']
    lon = df['lon']
    result = fromGPStoUTM(lon, lat)

    # Process photos and save the resulting DataFrame to a CSV file
    processed_df = process_photos(df,path_img_dir, result)
    processed_df.to_csv(path_navinfo_csv, index=False)
    print("Processing complete. The new CSV file has been saved.")