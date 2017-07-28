# -*- coding: utf-8 -*-
from pysocialwatcher import watcherAPI
import pandas as pd
watcher = watcherAPI()
watcherAPI.load_credentials_file("fb_credentials.csv")
dataframe = watcherAPI.run_data_collection("countries_age_ranges_diabetes_obesity_smoking.json")