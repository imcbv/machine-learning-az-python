#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:55:35 2017

@author: imcbv
"""

# Random Selection

# Reset variables
%reset -f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing random selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
# Visualising result on Histogram
plt.hist(ads_selected)
plt.title("Histogram of ads selction")
plt.xlabel("Ad index")
plt.ylabel("Time")
plt.show()