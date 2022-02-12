import pandas as pd
import numpy as np
import math
import random

# Set number of records that should be simulated
NROWS = 10000

# Set seed value to recreate same simulation
random.seed(3901531835)
np.random.seed(4028513893)

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
df['economic_orientation'] = np.random.uniform(size=NROWS)
df['economic_orientation'] = df['economic_orientation'] + \
    df['income'].apply(math.log)
s_min = df['economic_orientation'].min()
s_max = df['economic_orientation'].max()
df['economic_orientation'] = df['economic_orientation'].apply(
    lambda x: round((5*(x - s_min))/(s_max - s_min)))


# VARIABLE: PaidMembership
df['paid_membership'] = np.random.uniform(size=NROWS)
w1, w2, w3, w4, w5 = 1.5, -1.4, 1.8, 0.00004, 0.2
df['paid_membership'] = df['paid_membership'] + df['program_attractiveness']*w1 + \
    df['switching_costs']*w2 + df['quality_perception'] * \
    w3 + df['income']*w4 + df['economic_orientation']*w5
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
def setZeroFreq(x, p):
    return x['purchase_freq'] * (np.random.random() < p)


df['purchase_freq'] = np.random.uniform(
    df['paid_membership']*3.0, 5.0, size=NROWS) + np.random.randint(-2, 2, size=NROWS)
df['purchase_freq'] = (df['purchase_freq'] +
                       df['economic_orientation'] / 3).apply(int)
df['purchase_freq'] = df['purchase_freq'].apply(lambda x: int(max(0, x)))
df['purchase_freq'] = df.apply(lambda x: setZeroFreq(
    x, 0.99) if x['paid_membership'] else setZeroFreq(x, 0.95), axis=1)

# VARIABLE: SOW
df['sow'] = np.random.lognormal(mean=0.5, sigma=0.4, size=NROWS)
w1, w2, w3, w4 = 0.7, 0.84, 2.9, 0.05
df['sow'] = (df['paid_membership']*w1 + df['economic_orientation']
             * w2 + df['satisfaction']*w3 + df['market_demand']*w4) * df['sow']
df['sow'] = df.apply(
    lambda x: x['sow'] if x['purchase_freq'] > 0 else 0, axis=1)
s_min = df['sow'].min()
s_max = df['sow'].max()
df['sow'] = df['sow'].apply(lambda x: round(
    ((100*(x - s_min))/(s_max - s_min))/5)*5) / 100

# VARIABLE: Revenue
df['revenue'] = np.random.uniform(1000, 5000, size=NROWS)
w1, w2 = 1500, 300
df['revenue'] = df['revenue'] * df['sow'].apply(math.exp) * df['market_demand'].apply(
    math.exp) + df['paid_membership']*w1 + df['purchase_freq']*w2
df['revenue'] = df.apply(lambda x: x['revenue']
                         if x['purchase_freq'] > 0 else 0, axis=1)
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
