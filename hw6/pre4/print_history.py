import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import os, sys, time, datetime, pickle
# initlize model
history = pickle.load(open(sys.argv[1],'rb'))
print( len(history['val_acc']) )
print( history['val_acc'] )
