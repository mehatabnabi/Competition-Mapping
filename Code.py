#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

print('PART A.1')

cars_od = pd.read_csv("cars.dissimilarity.csv", sep=",", index_col=0)
k_vals = [i for i in range(1, 10)]
stress_vals = [0 for i in range(len(k_vals))]

for i, k in enumerate(k_vals):
    mds = MDS(n_components=k, metric=True, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    mds_fit = mds.fit(cars_od)
    stress_vals[i] = mds_fit.stress_
    
plt.figure(figsize = (10,10))
plt.plot(k_vals, stress_vals)
plt.xlabel('k',fontsize = 10)
plt.ylabel('Stress',fontsize = 10)
plt.title('Stress Vs k',fontsize = 10)
plt.grid()
plt.show()



print('PART A.2')

np.random.seed(101)
# k = 2
mds = MDS(n_components=2, metric=True, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
mds_fit = mds.fit(cars_od)
coords = mds_fit.embedding_
labels = cars_od.columns
x_offset = 0.1
plt.figure(figsize = (10,10))
plt.scatter(coords[:,0], coords[:,1])
for i in range(len(coords)):
    x, y = coords[i][0] + x_offset, coords[i][1]
    plt.annotate(text=labels[i], xy=(x,y), fontsize = 12)
plt.axis('square')
plt.show()

print('------------------------------------------------------------------------------------------------------')
print('PART B.1')
cars_od = pd.read_csv("cars.dissimilarity.csv", sep=",", index_col=0)
np.random.seed(101)
k_vals = [i for i in range(1, 6)]
stress_vals = [0 for i in range(len(k_vals))]
for i, k in enumerate(k_vals):

    # metric=False
    mds = MDS(n_components=k, metric=False, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    mds_fit = mds.fit(cars_od)
    stress_vals[i] = mds_fit.stress_
plt.figure(figsize = (4,7))
plt.plot(k_vals, stress_vals)
plt.xlabel('k',fontsize = 10)
plt.ylabel('Stress',fontsize = 10)
plt.title('Stress Vs k',fontsize = 10)
plt.grid()
plt.show()

print()

print('PART B.2')

np.random.seed(101)
#metric=False and k = 2
mds = MDS(n_components=2, metric=False, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
mds_fit = mds.fit(cars_od)
coords = mds_fit.embedding_
labels = cars_od.columns
x_offset = 0.015
x_offset = 0.015
plt.figure(figsize = (10,10))
plt.scatter(coords[:,0], coords[:,1])
for i in range(len(coords)):
    x, y = coords[i][0] + x_offset, coords[i][1]
    plt.annotate(text=labels[i], xy=(x,y),fontsize = 12)
plt.axis('square')
plt.show()

print('------------------------------------------------------------------------------------------------------')
get_ipython().system('pip install factor_analyzer')
from factor_analyzer import FactorAnalyzer
print('PART C.1')
cars_ar = pd.read_csv("cars.ar.csv", sep=",", index_col=0)
n_factors = [i for i in range(1, 11)]
# list to store goodness of fit values
gof = [0 for i in range(len(n_factors))]
for i, n_factor in enumerate(n_factors):
    fa = FactorAnalyzer(n_factors=n_factor, rotation=None)
    fa_fit_out = fa.fit(cars_ar)
    # extract communalities
    fa_communalities = fa_fit_out.get_communalities()
    # extract goodness of fit
    fa_gof = sum(fa_communalities)
    gof[i] = fa_gof
    # extract scores
    fa_scores = fa_fit_out.transform(cars_ar)
    # extract loadings
    fa_factor_loadings = fa_fit_out.loadings_
plt.figure(figsize = (10,10))
plt.plot(n_factors, gof)
plt.xlabel('n_factors',fontsize = 12)
plt.ylabel('GOF',fontsize = 12)
plt.title('GOF vs. n_factors',fontsize = 12)
plt.grid()
plt.show()

print('PART C.2')
fa = FactorAnalyzer(n_factors=2, rotation=None)
fa_fit_out = fa.fit(cars_ar)
fa_communalities = fa_fit_out.get_communalities()
gof = sum(fa_communalities)
fa_scores = fa_fit_out.transform(cars_ar)
fa_factor_loadings = fa_fit_out.loadings_
x_coords = fa_fit_out.transform(cars_ar)[:,0]
y_coords = fa_fit_out.transform(cars_ar)[:,1]

x_part3 = fa_fit_out.transform(cars_ar)[:,0]
y_part3 = fa_fit_out.transform(cars_ar)[:,1]

fig, axs = plt.subplots(1,1,sharex=True, sharey=True, figsize=(15,15))

axs.scatter(x_coords,y_coords)
axs.grid(False)
axs.set_title('Distance Map')
axs.axis('square')
axs.set_ylim([-5,5])
axs.set_xlim([-5,5])

r_sq_val = fa_communalities
betax_vals = fa_factor_loadings[:,0]
betay_vals = fa_factor_loadings[:,1]
attributes = list(cars_ar.columns)
arrow_end_x_list = []
arrow_end_y_list = []


for i in range(15):
    
    R2value = r_sq_val[i]
    betax = betax_vals[i]
    betay = betay_vals[i]
    
    arrowlengthscaleup = 3
    arrow_origin_x, arrow_origin_y = 0,0
    arrow_end_x = (arrowlengthscaleup*R2value*betax)/(np.sqrt(betax**2+betay**22))
    arrow_end_y = (arrowlengthscaleup*R2value*betay)/(np.sqrt(betax**2+betay**2))
    arrow_end_x_list.append(arrow_end_x)
    arrow_end_y_list.append(arrow_end_y)
    y_offset = 0.5
    x_offset = -0.2
    axs.arrow(arrow_origin_x, arrow_origin_y, arrow_end_x-arrow_origin_x, arrow_end_y-arrow_origin_y, length_includes_head=True, head_width=0.08, head_length=0.0002)
    axs.annotate(attributes[i],(arrow_end_x,arrow_end_y),fontsize = 10)


names = list(cars_od.columns)
for i in range(len(names)):
    y_offset = 0.2
    axs.annotate(names[i],(x_coords[i],y_coords[i]), fontsize = 10)
    
# plt.savefig('C_2.png', dpi=600)
plt.show()


# In[ ]:




