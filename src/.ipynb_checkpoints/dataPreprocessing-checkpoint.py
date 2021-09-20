#!/usr/bin/env python

"""dataPreprocessing.py: Given a cleaned dataframe, builds label encodings for categorical variables, 
then scales the data and returns 2 dataframes containing scaled data and unscaled data, and a pandas series containing ids
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def dataPreprocessor(X):
    #Gender string is replaced with a dummy variable
    X['Is_Male'] = X['Gender']
    X.loc[X['Gender'] == 'M', 'Is_Male'] = 1
    X.loc[X['Is_Male'] != 1, 'Is_Male'] = 0
    
    #Card categories are consolidated into a premium/non-premium dummy variable due to a lack of premium cards in the dataset
    X['Is_Premium'] = X['Card_Category']
    X.loc[X['Card_Category'] == 'Blue', 'Is_Premium'] = 0
    X.loc[X['Is_Premium'] != 0, 'Is_Premium'] = 1

    #Income labels are manually encoded, since automatic encoders did not maintain an increasing order from one tier to the next
    X['Income_Tier'] = X['Income_Category']
    X.loc[X['Income_Category'] == 'Less than $40K', 'Income_Tier'] = 0
    X.loc[X['Income_Category'] == '$40K - $60K', 'Income_Tier'] = 1
    X.loc[X['Income_Category'] == '$60K - $80K', 'Income_Tier'] = 2
    X.loc[X['Income_Category'] == '$80K - $120K', 'Income_Tier'] = 3
    X.loc[X['Income_Category'] == '$120K +', 'Income_Tier'] = 4

    #New labels are stored as integers for compatibility with visualization software
    X = X.astype({'Is_Premium': 'int64', 'Income_Tier':'int64', 'Is_Male':'int64'})
    
    #store ids for later use
    ids = X['CLIENTNUM']

    #Drop categorical variables once they are encoded
    X = X.drop(['Income_Category', 'Card_Category', 'Gender', 'CLIENTNUM'], axis=1)
    
    #Scale data to ensure greater equality in the influence each attribute can have
    scaler1 = MinMaxScaler()
    cols = X.columns
    X_scaled = pd.DataFrame(scaler1.fit_transform(X))
    X_scaled.columns = cols
    
    
    return X_scaled, X, ids