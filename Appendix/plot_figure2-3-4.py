#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys
import random
import pickle
import shutil
import itertools
import collections
from collections import namedtuple
from collections import defaultdict
from operator import itemgetter
from matplotlib.font_manager import FontProperties
import matplotlib.mlab as mlab
import matplotlib.pyplot as pl
import scipy.stats as stats
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

user_list=[1]*6050
user_list_x=[1]*6050

movie_list=[1]*3958
movie_list_x=[1]*3958

path='data/ml-1m/ratings.dat'
sep='::'
for i in range(6049):
    user_list_x[i]=i
for i in range(3958):
    movie_list_x[i]=i


with open(path) as f:
    i=1
    for line in f:
        user,movie= line.strip('\r\n').split(sep)[:2]
        if i<10:
            i+=1
            print(user,movie)
        uid=int(user)
        mid=int(movie)
        user_list[uid]+=1
        if i<10:
            print("usr_list","usr_id=",uid,user_list[uid])
           
        movie_list[mid]+=1
user_list.sort(reverse=True)
movie_list.sort(reverse=True)


pl.hist(user_list[200:],bins=100,color='g',edgecolor='b',alpha=0.5)
pl.title('User Active')
pl.xlabel('USER')
pl.ylabel('RATING NUMBER')

pl.show()    
