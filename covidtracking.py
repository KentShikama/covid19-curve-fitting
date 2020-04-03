import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

ASSUMED_RATIO = 0.03

def model(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d

df = pd.read_json("https://covidtracking.com/api/states/daily").iloc[::-1]
all_states = df.groupby("state")["state"].nunique().keys().to_numpy()
national_initial_date = pd.to_datetime(df["date"].min(), format="%Y%m%d")
national_initial_date_as_int = national_initial_date.timestamp() / 86400
days = np.arange(1, 150)

state_results = []
for state_name in all_states:
    state = df[df["state"] == state_name]

    dates = pd.to_datetime(state["date"], format="%Y%m%d")
    dates_as_int = dates.astype(int) / 10 ** 9 / 86400
    dates_as_int_array = dates_as_int.to_numpy()
    dates_as_int_array_normalized = dates_as_int_array - dates_as_int_array[0]
    cases = state["positive"].to_numpy()
    normalized_cases = state["positive"] * ((state["positive"] / state["totalTestResults"]) / ASSUMED_RATIO)

    try:
        popt, pcov = curve_fit(model, dates_as_int_array_normalized, normalized_cases)

        plt.title(state_name)
        plt.scatter(dates_as_int_array_normalized, cases, label=f"Confirmed cases")
        plt.scatter(dates_as_int_array_normalized, normalized_cases, label=f"Confirmed cases \n(normalized to {round(ASSUMED_RATIO * 100)}% testing positive)")
        plt.plot(days, model(days, *popt), 'g--', label=f"Predicted true cases")
        plt.xlabel(f'# days since {dates.min()}')
        plt.legend(loc=4)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.savefig(f'covid_tracking_images/{state_name}.png', bbox_inches = 'tight')
        plt.close()

        initial_date_as_int = dates_as_int.min()
        offset = initial_date_as_int - national_initial_date_as_int
        state_result = np.concatenate((np.zeros(int(offset)), model(days, *popt)))
        state_results.append(state_result[:len(days)])
    except Exception as e:
        print(e)

df_national = df.groupby("date").sum()
national_cases = df_national["positive"].to_numpy()
normalized_national_cases = df_national["positive"] * ((df_national["positive"] / df_national["totalTestResults"]) / ASSUMED_RATIO)
days_so_far = np.arange(1, len(national_cases) + 1)
results = np.sum(state_results, axis=0)
plt.title("National")
plt.scatter(days_so_far, national_cases, label=f"Confirmed cases")
plt.scatter(days_so_far, normalized_national_cases, label=f"Confirmed cases \n(normalized to {round(ASSUMED_RATIO * 100)}% testing positive)")
plt.plot(days, results, 'g--', label=f"Predicted true cases")
plt.xlabel(f'# days since March 4th, 2020')
plt.legend(loc=4)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.savefig(f'covid_tracking_images/all.png', bbox_inches = 'tight')
plt.close()