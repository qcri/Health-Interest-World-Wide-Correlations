import pandas as pd
from glob import glob
import json
import sys
import ast
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import bayes_mvs
import xlsxwriter
import csv
import numpy as np

DIABETES_LABEL = "diabetes"
OBESITY_LABEL = "high_bmi"
SMOKING_LABEL = "smoking"
HIGH_GLUCOSE_LABEL = "high_glucose"

diabetes_interests = ["All diabetes","Diabetes mellitus type 1 awareness", "Diabetes mellitus type 2 awareness", "Diabetic diet", "Gestational diabetes", "Insulin resistance awareness", "Diabetic hypoglycemia", "Diabetes mellitus awareness", "Managing Diabetes", "Insulin index"]
obesity_interests = ["All obesity", "Obesity awareness", "Plus-size clothing", "Weight loss (Fitness And wellness)", "Dieting", "Bariatrics"]
smoking_interests = ["All Smoking","Smoking","Tobacco","Tobacco smoking","Lung cancer awareness","Cigarette","Hookah", "Smoking cessation"]

SUBJECT_CONDITIONS_AND_INTERESTS = {
    DIABETES_LABEL : diabetes_interests,
    HIGH_GLUCOSE_LABEL : diabetes_interests,
    SMOKING_LABEL : smoking_interests,
    OBESITY_LABEL : obesity_interests
}

placebo_interests = ["Placebo: Reading, Technology, Entertainment", "Placebo: Facebook"]

CODE_COLUMN_LABEL = "country_code"

LOCATION_LABEL_IHME = "Location"
LOCATION_IHME_COLUMN_IN_LOCATION_DATA = "ihme_name"
LOCATION_FACEBOOK_COLUMN_IN_LOCATION_DATA = "name"
ENGLISH_SPEAKERS_COLUMN_IN_LOCATION_DATA = "english"
LEAST_1MILLION_COLUMN_IN_LOCATION_DATA = "least1millionFBuser"
DEVELOPED_COUNTRIES_COLUMN_IN_LOCATION_DATA = "developed"

LOCATION_CODE_LABEL = "country_code"
LOCATION_NAME_LABEL = "Location Name"
GEOLOCATIONS_COLUMN_LABEL = "geo_locations"
VALUES_KEY = "values"
AGES_RANGES_FACEBOOK_COLUMN = "ages_ranges"
INTEREST_NAME_LABEL = "interest_name"
INTEREST_COLUMN_FACEBOOK = "interests"
AUDIENCE_COLUMN_FACEBOOK = "audience"
PREVALENCE_VALUE_IHME = "Value"

AAPearson = "Avg.Abs-Pearson"
AAPearsonPvalue = "Avg-pearson-pvalue"
AASpearman = "Avg.Abs-Spearman"
AASpearmanPvalue = "Avg-spearman-pvalue"

MeanPearson = "Mean-Pearson"
MeanPearsonStd = "Mean-Pearson-Std"
MeanPearsonLow = "Mean-Pearson-Low"
MeanPearsonHigh = "Mean-Pearson-High"
MeanPearsonPvalue = "Mean-pearson-pvalue"
MeanPearsonPvalueStd = "Mean-Pearson-pvalue-Std"
MeanPearsonPvalueLow = "Mean-pearson-pvalue-Low"
MeanPearsonPvalueHigh = "Mean-pearson-pvalue-High"

MeanSpearman = "Mean-Spearman"
MeanSpearmanStd = "Mean-Spearman-Std"
MeanSpearmanLow = "Mean-Spearman-Low"
MeanSpearmanHigh = "Mean-Spearman-High"
MeanSpearmanPvalue = "Mean-Spearman-pvalue"
MeanSpearmanPvalueStd = "Mean-Spearman-pvalue-Std"
MeanSpearmanPvalueLow = "Mean-spearman-pvalue-Low"
MeanSpearmanPvalueHigh = "Mean-spearman-pvalue-High"

ALPHA = 0.90
MAX_PEARSON_PVALUE = 0.1

COUNTRY_CATEGORY_KEY = "Country Category"
NUMBER_OF_ENTRIES = "Number of Entries"
summary_keys = [COUNTRY_CATEGORY_KEY, NUMBER_OF_ENTRIES, AAPearson, AAPearsonPvalue, AASpearman, AASpearmanPvalue, MeanPearson, MeanPearsonStd, MeanPearsonPvalue, MeanPearsonPvalueStd, MeanSpearman, MeanSpearmanStd, MeanSpearmanPvalue, MeanPearsonPvalueStd]

DISEASE_COLUMN = "disease"

AGE_LABEL_IHME = "Age"
AGE_MAX = "age_max"
AGE_MIN = "age_min"

locations_with_20s = ["ET", "RW", "HT", "ML", "MW", "SO", "BF", "BJ", "GN"]
locations_not_country = ["HK", "XK"]
locations_not_in_facebook = ["South Sudan","Republic of the Congo", "Democratic Republic of the Congo", "Congo", "Sudan", "Cote d'Ivoire", "Antigua and Barbuda","American Samoa",
    "Antigua and Barbuda",
    "Bhutan",
    "Cote d'Ivoire",
    "Cuba",
    "East Midlands",
    "East of England",
    "England",
    "Greater London",
    "Greenland",
    "Guam",
    "Iran",
    "North East England",
    "North West England",
    "Northern Ireland",
    "Northern Mariana Islands",
    "Palestine",
    "Puerto Rico",
    "Scotland",
    "South East England",
    "South Sudan",
    "South West England",
    "Stockholm",
    "Sudan",
    "Sweden except Stockholm",
    "Syria",
    "Virgin Islands, U.S.",
    "Wales",
    "West Midlands",
    "Yorkshire and the Humber",
    "North Korea"
]
locations_to_remove = locations_with_20s + locations_not_country + locations_not_in_facebook
locations_data = pd.read_csv("./countries_data/country_data.csv")

age_range_key = "Age Range"
prevalence_key = "Prevalence (%)"
audience_key = "Audience"
relative_audience_key = "Relative Audience"
audience_without_interest_key = "Audience Without Interest"
correlation_row_example = {}
correlation_row_example[age_range_key] = None
correlation_row_example[prevalence_key] = None
correlation_row_example[audience_key] = None
correlation_row_example[relative_audience_key] = None

class CorrelationData():
    def __init__(self, correlation_rows, pearson_corr, pearson_pvalue, spearman_corr, spearman_pvalue, interest, disease, location_name, location_code):
        self.has20s = False
        for corr in correlation_rows:
            if corr[audience_key] == 0:
                self.has20s = True
                break
        self.rows = correlation_rows
        self.pearson = pearson_corr
        self.pearson_pvalue = pearson_pvalue
        self.spearman = spearman_corr
        self.spearman_pvalue = spearman_pvalue
        self.interest = interest
        self.disease = disease
        self.location_name = location_name
        self.location_code = location_code


def remove_unwanted_countries_from_ihme(dataframe):
    return dataframe[~dataframe[LOCATION_LABEL_IHME].isin(locations_to_remove)]


def load_dataframe_from_folder(label):
    path = "../" + label + "/"
    list_files = glob(path + "*.csv")
    folder_dataframe = pd.DataFrame()
    for file_path in list_files:
        file_dataframe = pd.read_csv(file_path)
        file_dataframe = remove_unwanted_cells(file_dataframe)
        file_dataframe = remove_unwanted_countries_from_ihme(file_dataframe)
        file_dataframe = file_dataframe[~pd.isnull(file_dataframe[LOCATION_LABEL_IHME])]
        file_dataframe = add_facebook_location_code(file_dataframe)
        file_dataframe = add_min_max_age_columns_for_groundtruth(file_dataframe)
        file_dataframe[DISEASE_COLUMN] = label
        folder_dataframe = folder_dataframe.append(file_dataframe)
    folder_dataframe = folder_dataframe[folder_dataframe[CODE_COLUMN_LABEL].isin(["DZ","BH","EG","IQ","JO","KW","LB","LY","MA","OM","QA","SA","SO","TN","AE","YE","PS"])] #Remove non arabic countries
    return folder_dataframe


def is_unwanted_cell(cell):
    cell_as_text = str(cell)
    if "Global Burden of Disease Study 2015" in cell_as_text or "Available from" in cell_as_text or "For terms and conditions" in cell_as_text:
        return True
    return False

def get_age_groundtruth(row, type):
    if type == AGE_MAX:
        age = int(row[AGE_LABEL_IHME].split(" ")[0].split("-")[1])
    elif type == AGE_MIN:
        age = int(row[AGE_LABEL_IHME].split(" ")[0].split("-")[0])
    else:
        raise Exception
    return age

def get_age_facebook(row, type):
    age_range = ast.literal_eval(row[AGES_RANGES_FACEBOOK_COLUMN])
    if type == AGE_MAX:
        if "max" in age_range:
            age = age_range["max"]
        else:
            age = None
    elif type == AGE_MIN:
        age = age_range["min"]
    else:
        raise Exception
    return age


def add_min_max_age_columns_for_groundtruth(dataframe):
    age_max_column = dataframe.apply(get_age_groundtruth, args=[AGE_MAX], axis=1)
    age_min_column = dataframe.apply(get_age_groundtruth, args=[AGE_MIN], axis=1)
    dataframe[AGE_MAX] = age_max_column
    dataframe[AGE_MIN] = age_min_column
    return dataframe

def add_min_max_age_columns_for_facebook(dataframe):
    age_max_column = dataframe.apply(get_age_facebook, args=[AGE_MAX], axis=1)
    age_min_column = dataframe.apply(get_age_facebook, args=[AGE_MIN], axis=1)
    dataframe[AGE_MAX] = age_max_column
    dataframe[AGE_MIN] = age_min_column
    return dataframe


def remove_unwanted_cells(dataframe):
    dataframe = dataframe.applymap(lambda cell: None if is_unwanted_cell(cell) else cell)
    return dataframe


def get_location_code_given_row(row):
    location = row[LOCATION_LABEL_IHME]
    query_result_dataframe = locations_data[locations_data[LOCATION_IHME_COLUMN_IN_LOCATION_DATA] == location]
    if len(query_result_dataframe) != 1:
        print row
        raise Exception
    return query_result_dataframe.iloc[0][LOCATION_CODE_LABEL]


def add_facebook_location_code(dataframe):
    code_column = dataframe.apply(get_location_code_given_row, axis=1)
    dataframe[CODE_COLUMN_LABEL] = code_column
    return dataframe


def load_ground_truth_data():
    all_diseases_dataframe = pd.DataFrame()
    diabetes_dataframe = load_dataframe_from_folder(DIABETES_LABEL)
    high_glucose_dataframe = load_dataframe_from_folder(HIGH_GLUCOSE_LABEL)
    smoking_dataframe = load_dataframe_from_folder(SMOKING_LABEL)
    obesity_dataframe = load_dataframe_from_folder(OBESITY_LABEL)
    obesity_dataframe = obesity_dataframe[(obesity_dataframe[AGE_MIN] != 20) & (obesity_dataframe[AGE_MIN] != 45)]
    all_diseases_dataframe = all_diseases_dataframe.append(diabetes_dataframe)
    all_diseases_dataframe = all_diseases_dataframe.append(obesity_dataframe)
    all_diseases_dataframe = all_diseases_dataframe.append(smoking_dataframe)
    all_diseases_dataframe = all_diseases_dataframe.append(high_glucose_dataframe)
    all_diseases_dataframe = all_diseases_dataframe.sort_values(by=AGE_MIN)
    return all_diseases_dataframe


def build_correlation_rows(facebook_data, groundtruth_data, interest, location_code, disease):
    fbdata_with_interest = facebook_data[(facebook_data[LOCATION_CODE_LABEL] == location_code) & (facebook_data[INTEREST_NAME_LABEL] == interest)]
    fbdata_without_interest = facebook_data[(facebook_data[LOCATION_CODE_LABEL] == location_code) & pd.isnull(facebook_data[INTEREST_NAME_LABEL])]
    groundtruth_data_disease = groundtruth_data[(groundtruth_data[DISEASE_COLUMN] == disease) & (groundtruth_data[LOCATION_CODE_LABEL] == location_code)]
    min_ages = groundtruth_data_disease[AGE_MIN].unique()
    correlation_rows = []
    for min_age in min_ages:
        max_age = groundtruth_data_disease[groundtruth_data_disease[AGE_MIN] == min_age].iloc[0][AGE_MAX]
        fb_with_interest_row = fbdata_with_interest[(fbdata_with_interest[AGE_MIN] == min_age) & (fbdata_with_interest[AGE_MAX] == max_age)]
        fb_without_interest_row = fbdata_without_interest[(fbdata_without_interest[AGE_MIN] == min_age) & (fbdata_without_interest[AGE_MAX] == max_age)]
        gt_row = groundtruth_data_disease[(groundtruth_data_disease[AGE_MIN] == min_age) & (groundtruth_data_disease[AGE_MAX] == max_age)]
        if len(fb_with_interest_row) != 1 and len(fb_without_interest_row) != 1 and len(gt_row) != 1:
            raise Exception
        fb_with_interest_row = fb_with_interest_row.iloc[0]
        fb_without_interest_row = fb_without_interest_row.iloc[0]
        gt_row = gt_row.iloc[0]

        audience = fb_with_interest_row[AUDIENCE_COLUMN_FACEBOOK]
        audience_without_interest = fb_without_interest_row[AUDIENCE_COLUMN_FACEBOOK]
        if audience == 20:
            print "Audience == 0", location_code, min_age, max_age, interest
            audience = 0
        if audience_without_interest == 20:
            raise Exception

        relative_audience = audience / float(audience_without_interest)
        prevalence = gt_row[PREVALENCE_VALUE_IHME]
        correlation_row = correlation_row_example.copy()
        correlation_row[age_range_key] = str(min_age) + "-" + str(max_age)
        correlation_row[audience_key] = audience
        correlation_row[audience_without_interest_key] = audience_without_interest
        correlation_row[relative_audience_key] = relative_audience
        correlation_row[prevalence_key] = prevalence
        correlation_rows.append(correlation_row)
    return correlation_rows


def compute_correlation(correlations_rows):
    groundtruth_array = map(lambda row: row[prevalence_key], correlations_rows)
    fb_array = map(lambda row: row[relative_audience_key], correlations_rows)
    pearson, pearson_pvalue = pearsonr(groundtruth_array, fb_array)
    spearman, spearman_pvalue = spearmanr(groundtruth_array, fb_array)
    return (pearson, pearson_pvalue, spearman, spearman_pvalue)

def get_correlation_data(facebook_data, groundtruth_data, interest, location_code, disease):
    print interest, location_code
    location_name = facebook_data[facebook_data[LOCATION_CODE_LABEL] == location_code].iloc[0][LOCATION_NAME_LABEL]
    correlation_rows = build_correlation_rows(facebook_data, groundtruth_data, interest, location_code, disease)
    pearson, pearson_pvalue, spearman, spearman_pvalue = compute_correlation(correlation_rows)
    correlation = CorrelationData(correlation_rows, pearson, pearson_pvalue, spearman, spearman_pvalue, interest, disease, location_name, location_code)
    return correlation

def build_excel_output(correlations):
    pass

def run_correlations(facebook_data, groundtruth_data):
    fb_locations = facebook_data[CODE_COLUMN_LABEL].unique()
    correlations = []
    for subject_condition in SUBJECT_CONDITIONS_AND_INTERESTS.keys():
        for interest in SUBJECT_CONDITIONS_AND_INTERESTS[subject_condition]:
            for location_code in fb_locations:
                correlations.append(get_correlation_data(facebook_data, groundtruth_data, interest, location_code, subject_condition))

    for interest in placebo_interests:
        for subject_condition in SUBJECT_CONDITIONS_AND_INTERESTS.keys():
            for location_code in fb_locations:
                correlations.append(get_correlation_data(facebook_data, groundtruth_data, interest, location_code, subject_condition))

    return correlations

def get_location_code_from_facebook_row(row):
    geo_locations = row[GEOLOCATIONS_COLUMN_LABEL]
    geo_locations = ast.literal_eval(geo_locations)
    code = str(geo_locations[VALUES_KEY][0])
    return code

def get_interest_name(row):
    interest_name = None
    if not pd.isnull(row[INTEREST_COLUMN_FACEBOOK]):
        interest_value = ast.literal_eval(row[INTEREST_COLUMN_FACEBOOK])
        interest_name = interest_value["name"]
    return interest_name

def add_interest_name_column(dataframe):
    dataframe[INTEREST_NAME_LABEL] = dataframe.apply(get_interest_name, axis=1)
    return  dataframe

def get_location_name_from_row(row):
    location_code = row[LOCATION_CODE_LABEL]
    name = locations_data[locations_data[LOCATION_CODE_LABEL] == location_code].iloc[0][LOCATION_FACEBOOK_COLUMN_IN_LOCATION_DATA]
    return name

def add_location_name(dataframe):
    dataframe[LOCATION_NAME_LABEL] = dataframe.apply(get_location_name_from_row, axis=1)
    return dataframe

def load_facebook_data(path):
    dataframe = pd.read_csv(path)
    dataframe[LOCATION_CODE_LABEL] = dataframe.apply(get_location_code_from_facebook_row, axis=1)
    dataframe = add_interest_name_column(dataframe)
    dataframe = add_min_max_age_columns_for_facebook(dataframe)
    dataframe = add_location_name(dataframe)
    dataframe = dataframe.sort_values(by=AGE_MIN)
    dataframe = dataframe.sort_values(by=LOCATION_NAME_LABEL)
    dataframe = dataframe[~dataframe[LOCATION_CODE_LABEL].isin(locations_to_remove)]
    dataframe = dataframe[~dataframe[LOCATION_NAME_LABEL].isin(locations_to_remove)]
    dataframe = dataframe[dataframe[CODE_COLUMN_LABEL].isin(["DZ", "BH", "EG", "IQ", "JO", "KW", "LB", "LY", "MA", "OM", "QA", "SA", "SO", "TN", "AE", "YE", "PS"])]  # Remove non arabic countries
    return dataframe

def filter_correlations_given_interest_disease(correlations, interest, disease):
    filtered_correlations = filter(lambda corr: corr.interest == interest and corr.disease == disease, correlations)
    return filtered_correlations

def write_correlations_sheet(sheet, correlations, bold):
    row = 0
    col = 0
    sheet.write(row, col, correlations[0].interest + "-" + correlations[0].disease, bold)
    row += 1
    for correlation in correlations:
        sheet.write(row, col, correlation.location_name, bold)
        row += 1
        sheet.write(row, col, age_range_key, bold)
        sheet.write(row, col + 1, prevalence_key, bold)
        sheet.write(row, col + 2, audience_key, bold)
        sheet.write(row, col + 3, relative_audience_key, bold)
        row += 1
        for corr_row in correlation.rows:
            sheet.write(row, col, corr_row[age_range_key])
            sheet.write(row, col + 1, corr_row[prevalence_key])
            sheet.write(row, col + 2, corr_row[audience_key])
            sheet.write(row, col + 3, corr_row[relative_audience_key])
            row += 1
        sheet.write(row, col, "Pearson", bold)
        sheet.write(row, col + 1, "p-value", bold)
        sheet.write(row, col + 2, "Spearman", bold)
        sheet.write(row, col + 3, "p-value", bold)
        row += 1
        if pd.isnull(correlation.pearson):
            print "NaN correlations: ", correlation.interest, correlation.location_name
            correlation.pearson = None
            correlation.spearman = None
            correlation.pearson_pvalue = None
            correlation.spearman_pvalue = None
        sheet.write(row, col, correlation.pearson)
        sheet.write(row, col + 1, correlation.pearson_pvalue)
        sheet.write(row, col + 2, correlation.spearman)
        sheet.write(row, col + 3, correlation.spearman_pvalue)
        row += 2

def write_summary_header(row, sheet, bold):
    col = 0
    for key in summary_keys:
        sheet.write(row, col, key, bold)
        col += 1

def get_countries_categories():
    all_countries = locations_data[~pd.isnull(locations_data[LOCATION_IHME_COLUMN_IN_LOCATION_DATA])][LOCATION_CODE_LABEL].unique().tolist()
    locations_data[DEVELOPED_COUNTRIES_COLUMN_IN_LOCATION_DATA] = locations_data[DEVELOPED_COUNTRIES_COLUMN_IN_LOCATION_DATA].fillna(False)
    locations_data[ENGLISH_SPEAKERS_COLUMN_IN_LOCATION_DATA] = locations_data[ENGLISH_SPEAKERS_COLUMN_IN_LOCATION_DATA].fillna(False)
    developed = locations_data[locations_data[DEVELOPED_COUNTRIES_COLUMN_IN_LOCATION_DATA]][LOCATION_CODE_LABEL].unique().tolist()
    english = locations_data[locations_data[ENGLISH_SPEAKERS_COLUMN_IN_LOCATION_DATA]][LOCATION_CODE_LABEL].unique().tolist()
    return {
        "All Countries" : all_countries
    }

def perform_division_NAN_for_invalid(a, b):
    try:
        return a / float(b)
    except ZeroDivisionError:
        return np.nan


def get_correlations_absolute_average_summary(filtered_correlations):
    absolute_average_summary = {}
    abs_pearson = []
    abs_pearson_pvalue = []
    abs_spearman = []
    abs_spearman_pvalue = []

    for correlation in filtered_correlations:
        if pd.isnull(correlation.pearson) or pd.isnull(correlation.spearman):
            raise Exception
        abs_pearson.append(abs(correlation.pearson))
        abs_pearson_pvalue.append(abs(correlation.pearson_pvalue))
        abs_spearman.append(abs(correlation.spearman))
        abs_spearman_pvalue.append(abs(correlation.spearman_pvalue))

    absolute_average_summary[AAPearson]= perform_division_NAN_for_invalid(sum(abs_pearson), len(abs_pearson))
    absolute_average_summary[AAPearsonPvalue] = perform_division_NAN_for_invalid(sum(abs_pearson_pvalue), len(abs_pearson_pvalue))
    absolute_average_summary[AASpearman] = perform_division_NAN_for_invalid(sum(abs_spearman), len(abs_spearman))
    absolute_average_summary[AASpearmanPvalue] = perform_division_NAN_for_invalid(sum(abs_spearman_pvalue), len(abs_spearman_pvalue))
    return absolute_average_summary


def get_mean_and_confidence_summary(correlations):
    pearsons = map(lambda corr: corr.pearson, correlations)
    spearmans = map(lambda corr: corr.spearman, correlations)
    pearson_pvalues = map(lambda corr: corr.pearson_pvalue, correlations)
    spearman_pvalues = map(lambda corr: corr.spearman_pvalue, correlations)
    if len(correlations) > 1:
        spearman_mean, spearman_var, spearman_std = bayes_mvs(spearmans, alpha=ALPHA)
        pearson_mean, pearson_var, pearson_std = bayes_mvs(pearsons, alpha=ALPHA)
        pearson_pvalue_mean, pearson_pvalue_var, pearson_pvalue_std = bayes_mvs(pearson_pvalues, alpha=ALPHA)
        spearman_pvalue_mean, spearman_pvalue_var, spearman_pvalue_std = bayes_mvs(spearman_pvalues, alpha=ALPHA)

        mean_summary = { MeanPearson: pearson_mean[0],
                     MeanPearsonStd: pearson_std[0],
                     MeanPearsonPvalue: pearson_pvalue_mean[0],
                     MeanPearsonPvalueStd: pearson_pvalue_std[0],
                     MeanSpearman: spearman_mean[0],
                     MeanSpearmanStd: spearman_std[0],
                     MeanSpearmanPvalue: spearman_pvalue_mean[0],
                     MeanSpearmanPvalueStd: spearman_pvalue_std[0]}
    else:
        mean_summary = {MeanPearson: None,
                        MeanPearsonLow: None,
                        MeanPearsonHigh: None,
                        MeanPearsonStd:None,
                        MeanPearsonPvalue: None,
                        MeanPearsonPvalueLow: None,
                        MeanPearsonPvalueHigh: None,
                        MeanPearsonPvalueStd: None,
                        MeanSpearman: None,
                        MeanSpearmanLow: None,
                        MeanSpearmanHigh: None,
                        MeanSpearmanStd: None,
                        MeanSpearmanPvalue: None,
                        MeanSpearmanPvalueLow: None,
                        MeanSpearmanPvalueHigh: None,
                        MeanSpearmanPvalueStd: None,
                        }
    return mean_summary

def get_correlations_summary_line(correlations,countries_list,country_category, interest, disease):
    summary = {}
    summary[COUNTRY_CATEGORY_KEY] = country_category
    filtered_correlations = filter(lambda corr: (corr.location_code in countries_list) and corr.interest == interest and corr.disease == disease and corr.has20s == False and corr.pearson_pvalue <= MAX_PEARSON_PVALUE, correlations)
    summary[NUMBER_OF_ENTRIES] = len(filtered_correlations)
    summary.update(get_correlations_absolute_average_summary(filtered_correlations))
    summary.update(get_mean_and_confidence_summary(filtered_correlations))
    return summary


def write_summary(row, sheet, summary):
    col = 0
    for key in summary_keys:
        value = summary[key]
        if pd.isnull(value):
            value = str(value)
        if value == np.inf:
            value = str(value)
        sheet.write(row, col, value)
        col += 1

def write_correlations_summary(row, col, sheet, correlations, bold):
    countries_categories = get_countries_categories()

    for disease_label in SUBJECT_CONDITIONS_AND_INTERESTS.keys():
        interests = SUBJECT_CONDITIONS_AND_INTERESTS[disease_label] + placebo_interests
        for interest in interests:
            row += 1
            sheet.write(row, col, interest + "-" + disease_label , bold)
            row += 1
            write_summary_header(row, sheet, bold)
            row += 1
            for country_category in countries_categories:
                countries_list = countries_categories[country_category]
                summary = get_correlations_summary_line(correlations, countries_list, country_category, interest, disease_label)
                write_summary(row, sheet, summary)
                row += 1
    return row+1


def write_summary_sheet(sheet, correlations, bold):
    row = 0
    col = 0
    row = write_correlations_summary(row, col, sheet, correlations, bold)
    # WRITE INTERESTS SUMMARY
    sheet.write(row, col, "INTERESTS USED", bold)
    row += 1
    for subject_condition_index in range(0,len(SUBJECT_CONDITIONS_AND_INTERESTS.keys())):
        subject_condition = SUBJECT_CONDITIONS_AND_INTERESTS.keys()[subject_condition_index]
        sheet.write(row, col + subject_condition_index, subject_condition, bold)
    row += 1
    for subject_condition in SUBJECT_CONDITIONS_AND_INTERESTS.keys():
        subject_condition_row = row
        max_row = 0
        interests = SUBJECT_CONDITIONS_AND_INTERESTS[subject_condition] + placebo_interests
        for interest in interests:
            sheet.write(subject_condition_row, col, interest)
            subject_condition_row += 1
            max_row = max(max_row, subject_condition_row)

    row = max_row
    # WRITE REMOVED LOCATIONS
    sheet.write(row, col, "REMOVED LOCATIONS", bold)
    row += 1
    sheet.write(row, col, "Fb Audience < 500k")
    sheet.write(row, col + 1, str(locations_not_country) + "not in IHME")
    sheet.write(row, col + 2, str(locations_with_20s) + " has 20s in at least 1 interest")
    sheet.write(row, col + 2, str(locations_not_in_facebook) + " not in facebook")
    row += 1

def post_process_correlations(correlations):
    workbook = xlsxwriter.Workbook('worldwide_correlations.xlsx')
    bold = workbook.add_format({'bold': True})
    worksheet = workbook.add_worksheet()
    worksheet.name = "Summary"
    write_summary_sheet(worksheet, correlations, bold)
    for subject_condition in SUBJECT_CONDITIONS_AND_INTERESTS.keys():
        interests = SUBJECT_CONDITIONS_AND_INTERESTS[subject_condition]
        for interest in interests:
            worksheet = workbook.add_worksheet()
            worksheet.name = subject_condition + "-" + interest[0:10]
            sheet_correlations = filter_correlations_given_interest_disease(correlations, interest, subject_condition)
            write_correlations_sheet(worksheet, sheet_correlations, bold)
    workbook.close()

def get_coutry_data_given_correlation(correlation):
    location_code = correlation.location_code
    query_dataframe = locations_data[locations_data[LOCATION_CODE_LABEL] == location_code]
    if len(query_dataframe) != 1:
        raise Exception
    return query_dataframe.iloc[0]

def generate_yelenas_csv_file(correlations):
    yelenas_dataframe = pd.DataFrame()
    rows = []
    statistic_key = "statistic"
    interest_key = "interest"
    country_name_key = "country_name"
    country_code_key = "country_code"
    is_english_key = "is_english"
    is_developed_key = "is_developed"
    ihme_prevalence_key = "ihme_prevalence"
    interest_audience_key = "interest_audience"
    yelenas_audience_without_interest_key = "audience_without_interest"
    interest_relative_audience_key = "interest_relative_audience"
    yelenas_age_range_key = "age_range"

    row_example = {
        statistic_key : None,
        interest_key: None,
        yelenas_age_range_key : None,
        country_name_key: None,
        country_code_key: None,
        is_english_key: None,
        is_developed_key: None,
        ihme_prevalence_key: None,
        interest_audience_key: None,
        interest_relative_audience_key: None,
        yelenas_audience_without_interest_key : None
    }

    for correlation in correlations:
        country_data = get_coutry_data_given_correlation(correlation)

        for correlation_row in correlation.rows:
            new_row = row_example.copy()
            new_row[statistic_key] = correlation.disease
            new_row[interest_key] = correlation.interest
            new_row[country_code_key] = country_data[LOCATION_CODE_LABEL]
            new_row[country_name_key] = country_data[LOCATION_IHME_COLUMN_IN_LOCATION_DATA]
            new_row[is_english_key] = country_data[ENGLISH_SPEAKERS_COLUMN_IN_LOCATION_DATA]
            new_row[is_developed_key] = country_data[DEVELOPED_COUNTRIES_COLUMN_IN_LOCATION_DATA]
            new_row[ihme_prevalence_key] = correlation_row[prevalence_key]
            new_row[interest_audience_key] = correlation_row[audience_key]
            new_row[interest_relative_audience_key] = correlation_row[relative_audience_key]
            new_row[yelenas_audience_without_interest_key] = correlation_row[audience_without_interest_key]
            new_row[yelenas_age_range_key] = correlation_row[age_range_key]
            rows.append(new_row)
    pd.DataFrame(rows).to_csv("yelenas_format_data.csv")


if __name__ == '__main__':
    print "Loading datasets..."
    facebook_data = load_facebook_data("all_interests_facebook_data.csv")
    groundtruth_data = load_ground_truth_data()
    print "Running correlations..."
    correlations = run_correlations(facebook_data, groundtruth_data)
    # print "Building excel..."
    post_process_correlations(correlations)
    print "Generating Yelenas csv file..."
    generate_yelenas_csv_file(correlations)



