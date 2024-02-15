#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pickle
sns.set(font_scale=3)
sns.set_style(style='white')
from matplotlib.ticker import FuncFormatter


#%%
def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:.2f}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str
# %%
adult_all = pd.read_csv("adult_allNN.csv")
adult_one = pd.read_csv("adult_1NN.csv")
adult_two = pd.read_csv("adult_2NN.csv")

german_all = pd.read_csv("cc_NN_all.csv")
german_one = pd.read_csv("cc_NN_1.csv")
german_two = pd.read_csv("cc_NN_2.csv")

law_all = pd.read_csv("law_rf_all.csv")
law_one = pd.read_csv("law_rf_1.csv")
law_two = pd.read_csv("law_rf_2.csv")

compas_all = pd.read_csv("compas_rf_all.csv")
compas_one = pd.read_csv("compas_rf_1.csv")
compas_two = pd.read_csv("compas_rf_2.csv")

german_dp_all = german_all['dp']
german_dp_one = german_one['dp']
german_dp_two = german_two['dp']

german_eq_all = german_all['eqop']
german_eq_one = german_one['eqop']
german_eq_two = german_two['eqop']

law_dp_all = law_all['dp']
law_dp_one = law_one['dp']
law_dp_two = law_two['dp']

law_eq_all = law_all['eqop']
law_eq_one = law_one['eqop']
law_eq_two = law_two['eqop']

compas_dp_all = compas_all['dp']
compas_dp_one = compas_one['dp']
compas_dp_two = compas_two['dp']

compas_eq_all = compas_all['eqop']
compas_eq_one = compas_one['eqop']
compas_eq_two = compas_two['eqop']

adult_dp_all = adult_all['dp']
adult_dp_one = adult_one['dp']
adult_dp_two = adult_two['dp']

adult_eq_all = abs(adult_all['eqop'])
adult_eq_one = adult_one['eqop']
adult_eq_two = adult_two['eqop']

german_accu_all = 1-german_all['accuracy']
german_accu_one = 1-german_one['accuracy']
german_accu_two = 1-german_two['accuracy']
adult_accu_all = adult_all['accuracy']
adult_accu_one = adult_one['accuracy']
adult_accu_two = adult_two['accuracy']
law_accu_all = law_all['accuracy']
law_accu_one = law_one['accuracy']
law_accu_two = law_two['accuracy']
compas_accu_all = compas_all['accuracy']
compas_accu_one = compas_one['accuracy']
compas_accu_two = compas_two['accuracy']

adult_dps = [adult_dp_all, adult_dp_one, adult_dp_two]
adult_eqs = [adult_eq_all, adult_eq_one, adult_eq_two]
adult_accus = [adult_accu_all, adult_accu_one, adult_accu_two]

compas_dps = [compas_dp_all, compas_dp_one, compas_dp_two]
compas_eqs = [compas_eq_all, compas_eq_one, compas_eq_two]
compas_accus = [compas_accu_all, compas_accu_one, compas_accu_two]

law_dps = [law_dp_all, law_dp_one, law_dp_two]
law_eqs = [law_eq_all, law_eq_one, law_eq_two]
law_accus = [law_accu_all, law_accu_one, law_accu_two]

german_dps = [german_dp_all, german_dp_one, german_dp_two]
german_eqs = [german_eq_all, german_eq_one, german_eq_two]
german_accus = [german_accu_all, german_accu_one, german_accu_two]
#%%

# %%
fig, ax = plt.subplots(4, 3,figsize = (25,20),sharex=True)
ax[0,0].boxplot(adult_accus)
ax[0,1].boxplot(adult_dps)
ax[0,2].boxplot(adult_eqs)

ax[1,0].boxplot(german_accus)
ax[1,1].boxplot(german_dps)
ax[1,2].boxplot(german_eqs)

ax[2,0].boxplot(law_accus)
ax[2,1].boxplot(law_dps)
ax[2,2].boxplot(law_eqs)

ax[3,0].boxplot(compas_accus)
ax[3,1].boxplot(compas_dps)
ax[3,2].boxplot(compas_eqs)

ax[0,0].set_title("Accuracy")
ax[0,1].set_title("Demographic Parity")
ax[0,2].set_title("Equality of Opportunity")
ax[0,0].set_ylabel("Adult (Gender)")
ax[1,0].set_ylabel("CC (Race)")
ax[2,0].set_ylabel("Law (Race)")
ax[3,0].set_ylabel("Compas (Race)")



major_formatter = FuncFormatter(my_formatter)
ax[0,0].yaxis.set_major_formatter(major_formatter)
ax[0,1].yaxis.set_major_formatter(major_formatter)
ax[0,2].yaxis.set_major_formatter(major_formatter)

ax[1,0].yaxis.set_major_formatter(major_formatter)
ax[1,1].yaxis.set_major_formatter(major_formatter)
ax[1,2].yaxis.set_major_formatter(major_formatter)


ax[0,0].xaxis.set_major_formatter(major_formatter)
ax[0,1].xaxis.set_major_formatter(major_formatter)
ax[0,2].xaxis.set_major_formatter(major_formatter)

ax[1,0].xaxis.set_major_formatter(major_formatter)
ax[1,1].xaxis.set_major_formatter(major_formatter)
ax[1,2].xaxis.set_major_formatter(major_formatter)

ax[2,0].yaxis.set_major_formatter(major_formatter)
ax[2,1].yaxis.set_major_formatter(major_formatter)
ax[2,2].yaxis.set_major_formatter(major_formatter)

ax[3,0].yaxis.set_major_formatter(major_formatter)
ax[3,1].yaxis.set_major_formatter(major_formatter)
ax[3,2].yaxis.set_major_formatter(major_formatter)

ax[2,0].xaxis.set_major_formatter(major_formatter)
ax[2,1].xaxis.set_major_formatter(major_formatter)
ax[2,2].xaxis.set_major_formatter(major_formatter)

ax[3,0].xaxis.set_major_formatter(major_formatter)
ax[3,1].xaxis.set_major_formatter(major_formatter)
ax[3,2].xaxis.set_major_formatter(major_formatter)
ax[3,0].set_xticks([0,1,2]) 
ax[3,0].set_xticklabels(['All features', '-1 -ve FairFIS feature', '-2 -ve FairFIS features'],rotation = 25)
ax[3,1].set_xticklabels(['All features', '-1 -ve FairFIS feature', '-2 -ve FairFIS features'],rotation = 25)
ax[3,2].set_xticklabels(['All features', '-1 -ve FairFIS feature', '-2 -ve FairFIS features'],rotation = 25)
#plt.savefig("visualization_rf.pdf", bbox_inches='tight')
# %%
