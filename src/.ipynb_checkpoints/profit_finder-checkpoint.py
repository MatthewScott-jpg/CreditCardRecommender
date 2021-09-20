#!/usr/bin/env python

"""This module creates an excel file containing individuals with high revenue potential"""

from data_loading import data_loader
from data_cleaning import data_cleaner
from data_preprocessing import data_preprocessor
from cluster_builder import cluster_builder
from outlier_locator import build_outlier_df

#print optimal dbscan esp value

def profit_finder():
    """Creates an excel file containing customers with potential for generating higher revenues

    Loops until a valid .csv file path is given
    """
    is_good_file = False
    file_input = input("Enter the csv file path: ")
    while is_good_file is False:
        try:
            with open(file_input):
                pass
            if file_input[-4:] == '.csv':
                is_good_file = True
            else:
                print("File is not in the .csv format")
                file_input = input("Enter the csv file path: ")
                continue
        except FileNotFoundError as e:
            print(e)
            file_input = input("Enter the csv file path: ")
            continue
    #'/Users/scott/Desktop/HCL/resources/BankChurners.csv'
    df = data_loader(file_input)
    print(f'File with {len(df)} observations loaded')
    cleaned_df = data_cleaner(df)
    if cleaned_df is None:
        return
    scaled_df, unscaled_df, ids = data_preprocessor(cleaned_df)
    labeled_df = cluster_builder(unscaled_df, scaled_df)
    outliers_df = build_outlier_df(labeled_df, ids)
    print(f'{len(outliers_df)} individuals located')
    outliers_df.to_excel("Identified_Customers.xlsx", index = False)
    print("List of individuals outputted to \"Identified_Customers.xlsx\"")

if __name__ == '__main__':
    profit_finder()
