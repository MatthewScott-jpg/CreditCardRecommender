#!/usr/bin/env python

"""This module selects and cleans attributes from a larger dataset"""

def data_cleaner(df):
    """Returns cleaned data

    Given a full dataframe, selects relevant attributes,
    cleans missing data, and returns a dataframe

    Args:
        df: A dataframe representing the full file input

    Returns:
        x: A cleaned dataframe

    Raises:
        KeyError: Key attribute missing in dataframe
    """
    try:
        x = df[['Attrition_Flag', 'Gender', 'Income_Category', 'Total_Trans_Ct','Card_Category',
       'Credit_Limit', 'Avg_Utilization_Ratio', 'CLIENTNUM']]
    except KeyError as e:
        print("Key attribute missing: ", e)
        return None

    x = x[x['Attrition_Flag'] == 'Existing Customer']
    x = x.drop(['Attrition_Flag'],axis=1)
    x = x[x['Income_Category'] != 'Unknown']

    return x
