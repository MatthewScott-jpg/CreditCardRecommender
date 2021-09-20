#!/usr/bin/env python

"""dataCleaning.py: Obtains desired attributes from the full dataframe, then removes missing data
and former customers. The analysis for what attributes are included is located in Reasearch.ipynb
"""

import pandas as pd

def dataCleaner(df):
    try:
        X = df[['Attrition_Flag', 'Gender', 'Income_Category', 'Total_Trans_Ct','Card_Category',
       'Credit_Limit', 'Avg_Utilization_Ratio', 'CLIENTNUM']]
    except KeyError:
        print("Key attribute missing")
        return

    X = X[X['Attrition_Flag'] == 'Existing Customer']
    X = X.drop(['Attrition_Flag'],axis=1)
    X = X[X['Income_Category'] != 'Unknown']
    
    return X
    
    
