#!/usr/bin/env python

"""This module loads a csv file into a dataframe"""

import pandas as pd

def data_loader(file):
    """Loads csv file into dataframe

    Args:
        file: a csv file

    Returns:
        df: a pandas dataframe containing the information in the inputed file
    """
    df = pd.read_csv(file)
    return df
