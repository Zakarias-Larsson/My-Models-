import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


df = pd.read_excel("C:\\Users\\Ägaren\\Min enhet\\Portfolio 2.0\\FactorModel\\FactorData.xlsx", sheet_name='M_Data')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# Step 2: Define custom rolling windows
custom_windows = [
    #("2016-09-30", "2021-08-31"),
    #("2016-10-31", "2021-09-30"),
    #("2016-11-30", "2021-10-31"),
    #("2016-12-31", "2021-11-30"),
    #("2017-01-31", "2021-12-31"),
   #("2017-02-28", "2022-01-31"),
    #("2017-03-31", "2022-02-28"),
    #("2017-04-30", "2022-03-31"),
    #("2017-05-31", "2022-04-30"),
    #("2017-06-30", "2022-05-31"),
    #("2017-07-31", "2022-06-30"),
    #("2017-08-31", "2022-07-31"),
    #("2017-09-30", "2022-08-31"),
    #("2017-10-31", "2022-09-30"),
    #("2017-11-30", "2022-10-31"),
    #("2017-12-31", "2022-11-30"),
    #("2018-01-31", "2022-12-31"),
    #("2018-02-28", "2023-01-31"),
    #("2018-03-31", "2023-02-28"),
    #("2018-04-30", "2023-03-31"),
    #("2018-05-31", "2023-04-30"),
    #("2018-06-30", "2023-05-31"),
    #("2018-07-31", "2023-06-30"),
    #("2018-08-31", "2023-07-31"),
    #("2018-09-30", "2023-08-31"),
    #("2018-10-31", "2023-09-30"),
    #("2018-11-30", "2023-10-31"),
    #("2018-12-31", "2023-11-30"),
    #("2019-01-31", "2023-12-31"),
    #("2019-02-28", "2024-01-31"),
    #("2019-03-31", "2024-02-29"),
    #("2019-04-30", "2024-03-31"),
    #("2019-05-31", "2024-04-30"),
    #("2019-06-30", "2024-05-31"),
    #("2019-07-31", "2024-06-30"),
    #("2019-08-31", "2024-07-31"),
    #("2019-09-30", "2024-08-31"),
    #("2019-10-31", "2024-09-30"),
    #("2019-11-30", "2024-10-31"),
    #("2019-12-31", "2024-11-30"),
    #("2020-01-31", "2024-12-31"),
    #("2020-02-29", "2025-01-31"),
    #("2020-03-31", "2025-02-28"),
    #("2020-04-30", "2025-03-31"),
    #("2020-01-02", "2025-03-31"),
    ("2024-05-14","2025-04-30")
]

# Step 3: Define regression function
def run_carhart_regression(window_df):
    Y = window_df['PLTR']
    X = window_df[['Mkt-RF', 'SMB', 'HML', 'MOM','RMW','CMA']]
    #X = window_df[['Mkt-RF', 'SMB', 'HML', 'MOM',]]
    #X = window_df[['Mkt-RF', 'SMB', ]]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return {
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "Prob (F-statistic)": model.f_pvalue,
        "Coefficients": model.params,
        "Standard Errors": model.bse,
        "t-values": model.tvalues,
        "p-values": model.pvalues,
        "Conf. Int.": model.conf_int(alpha=0.05),
        "Observations": int(model.nobs)
    }

# Step 4: Run regressions for each window
results = []
for start, end in custom_windows:
    window_df = df.loc[start:end].dropna()
    result = run_carhart_regression(window_df)
    result["Window"] = f"{start} to {end}"
    results.append(result)

# Step 5: Convert results into a dataframe
summary_data = []
for res in results:
    for factor in ['const', 'Mkt-RF', 'SMB', 'HML', 'MOM']:
        summary_data.append({
            "Window": res["Window"],
            "Factor": factor,
            "Coefficient": res["Coefficients"].get(factor, np.nan),
            "StdErr": res["Standard Errors"].get(factor, np.nan),
            "t-Value": res["t-values"].get(factor, np.nan),
            "p-Value": res["p-values"].get(factor, np.nan),
            "CI Lower": res["Conf. Int."].loc[factor][0] if factor in res["Conf. Int."].index else np.nan,
            "CI Upper": res["Conf. Int."].loc[factor][1] if factor in res["Conf. Int."].index else np.nan,
            "R-squared": res["R-squared"],
            "Adj. R-squared": res["Adj. R-squared"],
            "F-stat": res["F-statistic"],
            "F p-Value": res["Prob (F-statistic)"],
            "Obs": res["Observations"]
        })

summary_df = pd.DataFrame(summary_data)

# Loop through each rolling window result
for res in results:
    print(f"📅 Window: {res['Window']}")
    print(f"R-squared: {res['R-squared']:.4f}, Adjusted R-squared: {res['Adj. R-squared']:.4f}")
    print(f"F-statistic: {res['F-statistic']:.2f} (p = {res['Prob (F-statistic)']:.4g})")
    print("Coefficients:")
    print(res["Coefficients"].round(4))
    print("P-values:")
    print(res["p-values"].round(4))
    print("-" * 50)
