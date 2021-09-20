#!/usr/bin/env python

"""outlierLocator.py: Given a dataframe with labels and a series with ids, builds a new dataframe with customers that fall outside of normal ranges, and returns a dataframe with the customer's id number, cluster label, and a recommended strategy
"""

import pandas as pd

def locateLowUsePremium(X):
    customer_output = pd.DataFrame()
    
    premium_labels = [i for i in X.DBSCAN_Labels.unique() if X[X.DBSCAN_Labels==i].Is_Premium.mean() > 0]
    
    #isolate premium labels
    for i in premium_labels:
        tmp_df = X[X.DBSCAN_Labels==i]
        customer_output = pd.concat([customer_output,tmp_df[(tmp_df.Avg_Utilization_Ratio < tmp_df.Avg_Utilization_Ratio.mean()) &(tmp_df.Credit_Limit < tmp_df.Credit_Limit.mean()-tmp_df.Credit_Limit.std())][['ID','DBSCAN_Labels']]])
    return(customer_output)

def locatePremiumCandidates(X):
    customer_output = pd.DataFrame()
    
    nonpremium_labels = [i for i in X.DBSCAN_Labels.unique() if X[X.DBSCAN_Labels==i].Is_Premium.mean() < 1]
    
    #isolate nonpremium labels
    for i in nonpremium_labels:
        tmp_df = X[X.DBSCAN_Labels==i]
        customer_output = pd.concat([customer_output,tmp_df[(tmp_df.Income_Tier > tmp_df.Income_Tier.mean()) & (tmp_df.Total_Trans_Ct > tmp_df.Total_Trans_Ct.mean()+tmp_df.Total_Trans_Ct.std()) & (tmp_df.Avg_Utilization_Ratio > tmp_df.Avg_Utilization_Ratio.mean() + tmp_df.Avg_Utilization_Ratio.std())][['ID','DBSCAN_Labels']]])
    return(customer_output)
 
def buildOutlierDf(X, ids):
    X['ID'] = ids
    low_use_df = locateLowUsePremium(X)
    low_use_df['Recommendation'] = "Cashback Deals"
    potential_premium_df = locatePremiumCandidates(X)
    potential_premium_df['Recommendation'] = "Offer Premium"
    output_df = pd.concat([low_use_df,potential_premium_df])
    return output_df