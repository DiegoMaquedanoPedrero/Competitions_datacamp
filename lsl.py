import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Diego\Desktop\Competitions_datacamp\cosa.csv")
df = df.drop(["Unnamed: 0"], axis=1)
df = df[["DAYS_ENDDATE_FACT", "now_minus_first_application"]]

def cap_outliers_percentiles(df, vars_to_skip, perc1=5, perc2=95):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_result = df.copy()

    for c in list(df_result.columns):
        if (df_result[c].dtype in numerics) and (c not in vars_to_skip):
            # Computing 10th, 90th percentiles and replacing the outliers
            floor_percentile = np.nanpercentile(df_result[c], perc1)
            cap_percentile = np.nanpercentile(df_result[c], perc2)
            df_result[c] = np.where(df_result[c] > cap_percentile, cap_percentile, df_result[c])
            df_result[c] = np.where(df_result[c] < floor_percentile, floor_percentile, df_result[c])

    return df_result


loli = cap_outliers_percentiles(df, ["loan_id_now", "loan_id_past"])

loli["DAYS_ENDDATE_FACT"].describe()
df["DAYS_ENDDATE_FACT"].describe()


plt.margins(0.02)

"""
loli["now_minus_first_application"].describe()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
vars_to_skip = ["loan_id_now", "loan_id_past"]

for c in list(loli.columns):
    if (loli[c].dtype in numerics) and (c not in vars_to_skip):
        print(loli[c].describe())
"""
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["now_minus_first_application"])
plt.show()

_ = sns.ecdfplot(data=loli, x="now_minus_first_application")
plt.show()