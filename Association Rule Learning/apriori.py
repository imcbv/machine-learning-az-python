#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 01:19:48 2017

@author: imcbv
"""

# Apriori

# Reset variables
%reset -f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

# Print most relevant rule
results[0][0]