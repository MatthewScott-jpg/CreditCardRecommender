#!/usr/bin/env python

"""profitFinder.py: Using the analysis provided in the research environment as well as the functions contained in src,
the file builds and returns an excel file containing customers with potential for generating higher revenues
"""
import pandas as py

from dataLoading import dataLoader
from dataCleaning import dataCleaner
from dataPreprocessing import dataPreprocessor
from clusterBuilder import clusterBuilder
from outlierLocator import buildOutlierDf

def profitFinder():
    df = dataLoader('/Users/scott/Desktop/HCL/resources/BankChurners.csv')
    cleaned_df = dataCleaner(df)
    scaled_df, unscaled_df, ids = dataPreprocessor(cleaned_df)
    labeled_df = clusterBuilder(unscaled_df, scaled_df)
    outliers_df = buildOutlierDf(labeled_df, ids)
    print(outliers_df.head())
    
if __name__ == '__main__':
    profitFinder()