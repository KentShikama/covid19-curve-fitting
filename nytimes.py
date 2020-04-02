import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def model(x, a, b, c, d):
     return a*np.tanh(b*x+c)+d

df = pd.read_csv("us-states.csv")
all_states = df.groupby("state")["state"].nunique().keys().to_numpy()
national_initial_date = pd.to_datetime(df["date"].min(), format="%Y-%m-%d")
national_initial_date_as_int = national_initial_date.timestamp() / 86400
days = np.arange(1, 150)

state_results = []
for state_name in all_states:
    state = df[df["state"] == state_name]

    dates = pd.to_datetime(state["date"], format="%Y-%m-%d")
    dates_as_int = dates.astype(int) / 10**9 / 86400
    dates_as_int_array = dates_as_int.to_numpy()
    dates_as_int_array_normalized = dates_as_int_array - dates_as_int_array[0]
    cases = state["cases"].to_numpy()
    
    try:
        popt, pcov = curve_fit(model, dates_as_int_array_normalized, cases, p0=[500, 2, -3, 5])
    
        plt.title(state_name)
        plt.scatter(dates_as_int_array_normalized, cases)
        plt.plot(days, model(days, *popt), 'g--')
        plt.xlabel(f'# days since {dates.min()}')
        plt.ylabel(f'total cases')
        plt.savefig(f'nytimes/{state_name}.png')
        plt.close()
    
        initial_date_as_int = dates_as_int.min()
        offset = initial_date_as_int - national_initial_date_as_int
        state_result = np.concatenate((np.zeros(int(offset)), model(days, *popt)))
        state_results.append(state_result[:len(days)])
    except Exception as e:
        print(e)

national_cases = df.groupby("date").sum()["cases"].to_numpy()
days_so_far = np.arange(1, len(national_cases) + 1)
results = np.sum(state_results, axis=0)
plt.title("National")
plt.scatter(days_so_far, national_cases)
plt.plot(days, results, 'g--')
plt.xlabel(f'# days since 1/21/2020')
plt.ylabel(f'total cases')
plt.savefig(f'nytimes/all.png')
plt.close()

# popt, pcov = curve_fit(model, days_so_far, national_cases, p0=[500, 2, -3, 5])
# plt.plot(days, model(days, *popt), 'r--')
