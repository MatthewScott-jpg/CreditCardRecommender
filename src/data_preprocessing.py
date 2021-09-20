#!/usr/bin/env python

"""This module runs preprocessing on a cleaned dataframe"""

import pandas as pd
import sklearn.preprocessing as preprocessing

def data_preprocessor(x):
    """Returns a labeled dataframe, scaled dataframe, and id values

    Given a cleaned dataframe, builds label encodings for categorical variables,
    then scales the data and returns 2 dataframes containing scaled data and unscaled data,
    and a pandas series containing ids

    Args:
        x: A dataframe with relevant values selected by dataCleaner

    Returns:
        x: label encoded version of input x
        x_scaled: scaled version of label encoded x
        ids: series object containing customer id numbers
    """

    #Gender string is replaced with a dummy variable
    x['Is_Male'] = x['Gender']
    x.loc[x['Gender'] == 'M', 'Is_Male'] = 1
    x.loc[x['Is_Male'] != 1, 'Is_Male'] = 0

    #Card categories are consolidated into a premium/non-premium dummy variable
    #due to a lack of premium cards in the dataset
    x['Is_Premium'] = x['Card_Category']
    x.loc[x['Card_Category'] == 'Blue', 'Is_Premium'] = 0
    x.loc[x['Is_Premium'] != 0, 'Is_Premium'] = 1

    #Income labels are manually encoded,
    #since automatic encoders did not maintain an increasing order from one tier to the next
    x['Income_Tier'] = x['Income_Category']
    x.loc[x['Income_Category'] == 'Less than $40K', 'Income_Tier'] = 0
    x.loc[x['Income_Category'] == '$40K - $60K', 'Income_Tier'] = 1
    x.loc[x['Income_Category'] == '$60K - $80K', 'Income_Tier'] = 2
    x.loc[x['Income_Category'] == '$80K - $120K', 'Income_Tier'] = 3
    x.loc[x['Income_Category'] == '$120K +', 'Income_Tier'] = 4

    #New labels are stored as integers for compatibility with visualization software
    x = x.astype({'Is_Premium': 'int64', 'Income_Tier':'int64', 'Is_Male':'int64'})

    #store ids for later use
    ids = x['CLIENTNUM']

    #Drop categorical variables once they are encoded
    x = x.drop(['Income_Category', 'Card_Category', 'Gender', 'CLIENTNUM'], axis=1)

    #Scale data to ensure greater equality in the influence each attribute can have
    scaler1 = preprocessing.MinMaxScaler()
    cols = x.columns
    x_scaled = pd.DataFrame(scaler1.fit_transform(x))
    x_scaled.columns = cols

    return x_scaled, x, ids
