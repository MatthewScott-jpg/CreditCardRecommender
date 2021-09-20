#!/usr/bin/env python

"""dataLoading.py: Loads BankChurners csv file into pandas data"""

import pandas as pd

def dataLoader(file):
    df = pd.read_csv(file) 
    
    return df