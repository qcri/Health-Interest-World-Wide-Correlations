import pandas as pd
import numpy as np
from difflib import SequenceMatcher as sm
import argparse

countries_file_raw = "country_data_raw.csv"
countries_file_full = "country_data_full.csv"
countries_file_data = "country_data.csv"

def to_float(value):
  try:
    value = float(value)
    return value
  except ValueError:
    return np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process countries data')
    parser.add_argument('--raw', action="store_true", help='Process from raw')
    parser.add_argument('--post', action="store_true", help='Process from raw')
    args = parser.parse_args()
    if args.raw:
        countries_data_raw_df = pd.read_csv(countries_file_raw)

        #Begin With Country Code
        countries_data_df = countries_data_raw_df[countries_data_raw_df["field"] == "country_code"]
        countries_data_df = countries_data_df[["name","value"]]
        countries_data_df = countries_data_df.rename(columns={"value":"country_code"})
        #Fields to fill 
        fields = list(countries_data_raw_df["field"].drop_duplicates().values)

        fields.remove("country_code")

        for field in fields:
            print "Looking to field:", field
            field_df = countries_data_raw_df[countries_data_raw_df["field"] == field]
            countries_data_df[field] = np.nan
            for index, country_row in countries_data_df.iterrows():
                closest_row = [0,None]
                for index_field, field_row in field_df.iterrows():
                    similarity = sm(None,country_row["name"].lower(),field_row["name"].lower()).ratio()
                    if similarity > closest_row[0] and similarity > 0.9:
                        closest_row[0] = similarity
                        closest_row[1] = field_row
                
                if not closest_row[1] is None:
                    countries_data_df.loc[index,field] = closest_row[1].value
                    if closest_row[0] < 1:
                        print "CHECK IF MATCH:", country_row["name"], closest_row[1]["name"],closest_row[0], field
                else:
                    print "FILL:", field, country_row["name"]

        countries_data_df.to_csv(countries_file_full)
        print "Saved", countries_file_full

    if args.post:
        countries_data = pd.read_csv(countries_file_full)
        #Process WHO data
        for field in ["tobacco","high_glucose","obesity"]:
            countries_data[field] = countries_data[field].apply(lambda x:  to_float(str(x).split(" ")[0]) if x == x else np.nan)
        #Remove Countries we don't have population
        countries_data["gdp"] = countries_data["gdp"].apply(lambda x:  int(x) if x == x else np.nan)
        countries_data["oecd"] = countries_data.apply(lambda x: x["oecd"] == 1, axis=1)
        countries_data["english"] = countries_data.apply(lambda x: x["english_population"]/float(x["facebook_population"]) > 0.5, axis=1)
        countries_data["highfb"] = countries_data.apply(lambda x: x["facebook_population"]/float(x["country_population"]) > 0.3, axis=1)
        countries_data["highfb_english"] = countries_data.apply(lambda x: x["english"] & x["highfb"], axis=1)
        countries_data["least1millionFBuser"] = countries_data["facebook_population"].apply(lambda x: x > 1000000)

        for continent in ["asia","europe","africa","s_america","oceania","n_america"]:
            countries_data[continent] = countries_data["continent"].apply(lambda x: x == continent)

        # countries_data = countries_data[countries_data["tobacco"] == countries_data["tobacco"]] #Remove lines wihtou tobacco
        #Set Namibia country_code
        namibia_index = countries_data[countries_data["name"] == "Namibia"]["country_code"].index.values[0]
        countries_data.loc[namibia_index,"country_code"] = "NA"
        countries_data.to_csv(countries_file_data)
        print "Saved"