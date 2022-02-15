from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import math
import random

# Set number of records that should be simulated
NROWS = 10000

# Set seed value to recreate same simulation
random.seed(3901591835)
np.random.seed(4128513893)

# Initialize hidden variables
df = pd.DataFrame(np.random.uniform(size=(NROWS, 4)), columns=[
                  'program_attractiveness',
                  'switching_costs',
                  'market_demand',
                  'quality_perception'])


# VARIABLE: Income
df['income'] = np.random.randint(15000, 100000, NROWS)
df['income'] = df['income'].apply(lambda x: round(x/1000)*1000)


# VARIABLE: EconomicOrientation
df['economic_orientation'] = df['income'].apply(
    math.log) + np.random.uniform(size=NROWS)
s_min = df['economic_orientation'].min()
s_max = df['economic_orientation'].max()
df['economic_orientation'] = df['economic_orientation'].apply(
    lambda x: round((5*(x - s_min))/(s_max - s_min)))


# VARIABLE: PaidMembership
w1, w2, w3, w4, w5 = 1.5, -1.4, 1.8, 0.00004, 0.2
df['paid_membership'] = df['program_attractiveness']*w1 + df['switching_costs'] * \
    w2 + df['quality_perception'] * w3 + \
    df['income']*w4 + df['economic_orientation'] * \
    w5 + np.random.uniform(size=NROWS)
df['paid_membership'] = df['paid_membership'].apply(lambda x: int(x > 4))


# VARIABLE: Satisfaction
def f_satisfaction(x):
    if x['paid_membership']:
        y = random.choices([0, 1], [.1, .9])[0]
    else:
        y = random.choices([0, 1], [.9, .1])[0]
    return y + x['quality_perception'] + np.random.uniform()


df['satisfaction'] = df.apply(f_satisfaction, axis=1)
s_min = df['satisfaction'].min()
s_max = df['satisfaction'].max()
df['satisfaction'] = df['satisfaction'].apply(
    lambda x: round((x - s_min)/(s_max - s_min), 2))


# VARIABLE: PurchaseFreq
df['purchase_freq'] = (df['paid_membership']*4 + df['economic_orientation'] / 3 + np.random.uniform(
    df['paid_membership']*3.0, 5.0, size=NROWS) + np.random.randint(-2, 2, size=NROWS)).apply(int)
df['purchase_freq'] = df['purchase_freq'].apply(lambda x: int(max(0, x)))

# VARIABLE: SOW
w1, w2, w3, w4 = 0.7, 0.84, 2.9, 0.05
df['sow'] = (df['paid_membership']*w1 + df['economic_orientation'] * w2 + df['satisfaction']
             * w3 + df['market_demand']*w4) * np.random.lognormal(mean=0.5, sigma=0.4, size=NROWS)
df['sow'] = df.apply(
    lambda x: x['sow'] if x['purchase_freq'] > 0 else 0, axis=1)

s_min = df['sow'].min()
s_max = df['sow'].max()
df['sow'] = df['sow'].apply(lambda x: round(
    ((100*(x - s_min))/(s_max - s_min))/5)*5) / 100

# VARIABLE: Revenue
w1, w2 = 1500, 300
df['revenue'] = df['paid_membership']*w1 + df['purchase_freq']*w2 + df['sow'].apply(math.exp) * df['market_demand'].apply(
    math.exp) * np.random.uniform(1000, 5000, size=NROWS)
df['revenue'] = df.apply(lambda x: x['revenue']
                         if (x['purchase_freq'] + x['paid_membership']) > 0 else 0, axis=1)
df['revenue'] = df['revenue'].apply(lambda x: round(x/1000)*1000)


# Removing unobserved variables
df = df.drop(['program_attractiveness', 'switching_costs',
             'market_demand', 'quality_perception'], axis=1)

# Renaming variables
var_dict = {
    'income': "Income",
    'economic_orientation': "EconomicOrientation",
    'paid_membership': "PaidMembership",
    'satisfaction': "Satisfaction",
    'purchase_freq': "PurchaseFreq",
    'sow': "SOW",
    'revenue': "Revenue",
}
df = df.rename(mapper=var_dict, axis=1)

# Export simulated data
df.to_csv("paid_loyalty_data.csv", index=False)
