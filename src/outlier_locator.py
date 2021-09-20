#!/usr/bin/env python

"""Identifies recommended strategies for customers located outside of normal ranges"""

import pandas as pd

def locate_low_use_premium(x):
    """Locates under-utilization within premium cardholders

    Under-utilization is defined as individuals who have a utilization ratio below the mean and
    a credit limit more than one standard deviation below the mean, a group that analysis shows
    should have a high utilization ratio.

    Args:
        x: a preprocessed and labeled dataframe

    Returns:
        customer_output: a dataframe containing identified customer's ids and cluster labels
    """
    customer_output = pd.DataFrame()

    premium_labels = [i for i in x.DBSCAN_Labels.unique()
                      if x[x.DBSCAN_Labels==i].Is_Premium.mean() > 0]

    for i in premium_labels:
        tmp_df = x[x.DBSCAN_Labels==i]
        customer_output = pd.concat([customer_output,tmp_df[
            (tmp_df.Avg_Utilization_Ratio < tmp_df.Avg_Utilization_Ratio.mean())
            & (tmp_df.Credit_Limit < tmp_df.Credit_Limit.mean()-tmp_df.Credit_Limit.std())]
                                     [['ID','DBSCAN_Labels']]])
    return customer_output

def locate_premium_candidates(x):
    """Locates high interaction within non-premium cardholders

    High interaction is defined as individuals who are in a relatively high income tier
    in their cluster, a total transaction count more than one standard deviation above the mean,
    and a utilization ratio more than one standard deviation above the mean.

    Args:
        X: a preprocessed and labeled dataframe

    Returns:
        customer_output: a dataframe containing identified customer's ids and cluster labels
    """
    customer_output = pd.DataFrame()

    nonpremium_labels = [i for i in x.DBSCAN_Labels.unique()
                         if x[x.DBSCAN_Labels==i].Is_Premium.mean() < 1]

    #isolate nonpremium labels
    for i in nonpremium_labels:
        tmp_df = x[x.DBSCAN_Labels==i]
        customer_output = pd.concat([customer_output,tmp_df[
            (tmp_df.Income_Tier > tmp_df.Income_Tier.mean())
            & (tmp_df.Total_Trans_Ct > tmp_df.Total_Trans_Ct.mean()+tmp_df.Total_Trans_Ct.std())
            & (tmp_df.Avg_Utilization_Ratio >
               tmp_df.Avg_Utilization_Ratio.mean() + tmp_df.Avg_Utilization_Ratio.std())]
                                     [['ID','DBSCAN_Labels']]])
    return customer_output

def build_outlier_df(x, ids):
    """Identifies recommended strategies for located customers

    Uses premium and nonpremium locator functions to build a dataframe containing the
    desired customers with corresponding recommendations

    Args:
        x: a preprocessed and labeled dataframe
        ids: a series object containing customer ids

    Returns:
        output_df: a dataframe containing each identified customer's id, cluster label,
        and a recommendation
    """
    x['ID'] = ids
    x['Recommendation'] = "No Action"
    low_use_df = locate_low_use_premium(x)
    potential_premium_df = locate_premium_candidates(x)

    print(f'{len(low_use_df)+len(potential_premium_df)} potential individuals located')
    
    x['Low_Use'] = x.ID.isin(low_use_df.ID).astype(int)
    x.loc[x['Low_Use'] == 1, 'Recommendation'] = "Cashback Deals"
    
    x['New_Premium'] = x.ID.isin(potential_premium_df.ID).astype(int)
    x.loc[x['New_Premium'] == 1, 'Recommendation'] = "Offer Premium"
    
    x['Income_Label'] = ""
    x.loc[x['Income_Tier'] == 0, 'Income_Label'] = 'Less than $40K'
    x.loc[x['Income_Tier'] == 1, 'Income_Label'] = '$40K - $60K'
    x.loc[x['Income_Tier'] == 2, 'Income_Label'] = '$60K - $80K'
    x.loc[x['Income_Tier'] == 3, 'Income_Label'] = '$80K - $120K'
    x.loc[x['Income_Tier'] == 4, 'Income_Label'] = '$120K +'
    
    x['DBSCAN_Label'] = ""
    x.loc[x['DBSCAN_Labels'] == 0, 'DBSCAN_Label'] = 'Male, Non Premium'
    x.loc[x['DBSCAN_Labels'] == 1, 'DBSCAN_Label'] = 'Female, Non Premium'
    x.loc[x['DBSCAN_Labels'] == 2, 'DBSCAN_Label'] = 'Male, Premium'
    x.loc[x['DBSCAN_Labels'] == 3, 'DBSCAN_Label'] = 'Female, Premium'
    
    output_df = x[['ID','DBSCAN_Label','Income_Label','Avg_Utilization_Ratio','Total_Trans_Ct','Recommendation']]
    return output_df
