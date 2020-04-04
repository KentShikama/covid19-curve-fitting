import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

ASSUMED_RATIO = 0.03
DAYS_TO_PLOT = 150
LOOKBACK = 13

def model(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d

def run():
    df = pd.read_json("https://covidtracking.com/api/states/daily").iloc[::-1]
    all_states = df.groupby("state")["state"].nunique().keys().to_numpy()
    national_initial_date = pd.to_datetime(df["date"].min(), format="%Y%m%d")
    national_initial_date_as_int = national_initial_date.timestamp() / 86400
    days = np.arange(1, DAYS_TO_PLOT)

    ny_results_over_time = []
    results_over_time = []
    normalized_national_cases_over_time = []
    for lookback in range(LOOKBACK, -1, -1):
        state_results = []
        for state_name in all_states:
            state = df[df["state"] == state_name]
            state = state[:len(state) - lookback]

            dates = pd.to_datetime(state["date"], format="%Y%m%d")
            dates_as_int = dates.astype(int) / 10 ** 9 / 86400
            dates_as_int_array = dates_as_int.to_numpy()
            dates_as_int_array_normalized = dates_as_int_array - dates_as_int_array[0]
            cases = state["positive"].to_numpy()
            normalized_cases = compute_normalized_cases(state, state_name)

            try:
                popt, pcov = curve_fit(model, dates_as_int_array_normalized, normalized_cases)
                if lookback == 0:
                    plot_state_graph(cases, dates, dates_as_int_array_normalized, days, normalized_cases, popt, state_name)
                state_result = compute_state_result(dates_as_int, days, national_initial_date_as_int, popt)
                if state_name != "NJ":
                    state_results.append(state_result[:len(days)])
            except Exception as e:
                print(e)

        df_national = df.groupby("date").sum()
        df_national = df_national[:len(df_national) - lookback]
        national_cases = df_national["positive"].to_numpy()
        normalized_national_cases = df_national["positive"] * ((df_national["positive"] / df_national["totalTestResults"]) / ASSUMED_RATIO)
        days_so_far = np.arange(1, len(national_cases) + 1)
        results = np.sum(state_results, axis=0)
        if lookback == 0:
            plot_national_graph(days, days_so_far, national_cases, normalized_national_cases, results)
        normalized_national_cases_over_time.append(normalized_national_cases.iloc[-1])
        results_over_time.append(results)

    lookback_days = np.arange(0, LOOKBACK + 1)
    predicted_peaks = list(map(lambda g: g[-1], results_over_time))
    plt.title("Predicted peak case count nationally")
    plt.plot(lookback_days, predicted_peaks, 'g--', label=f"Predicted peak")
    plt.xlabel(f'Predictions over the past {LOOKBACK} days')
    plt.legend(loc=4)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.savefig(f'covid_tracking_images/predictions.png', bbox_inches='tight')
    plt.close()

    how_far_along_are_we = 100 * (np.array(normalized_national_cases_over_time) / np.array(predicted_peaks))
    plt.title("How far along are we?")
    plt.plot(lookback_days, how_far_along_are_we, label=f"% of predicted total cases confirmed \n(100% means pandemic is over)")
    plt.xlabel(f'Predictions over the past {LOOKBACK} days')
    plt.legend(loc=4)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.savefig(f'covid_tracking_images/how_far_along.png', bbox_inches='tight')
    plt.close()


def compute_normalized_cases(state, state_name):
    normalized_cases = state["positive"] * ((state["positive"] / state["totalTestResults"]) / ASSUMED_RATIO)
    if state_name == "WA":
        normalized_cases = pd.concat([state["positive"][:5], normalized_cases[5:]])
    return normalized_cases


def compute_state_result(dates_as_int, days, national_initial_date_as_int,
                         popt):
    initial_date_as_int = dates_as_int.min()
    offset = initial_date_as_int - national_initial_date_as_int
    state_result = np.concatenate((np.zeros(int(offset)), model(days, *popt)))
    return state_result


def plot_national_graph(days, days_so_far, national_cases,
                        normalized_national_cases, results):
    plt.title("National")
    plt.scatter(days_so_far, national_cases, label=f"Confirmed cases")
    plt.scatter(days_so_far, normalized_national_cases,
                label=f"Confirmed cases \n(normalized to {round(ASSUMED_RATIO * 100)}% testing positive)")
    plt.plot(days, results, 'g--', label=f"Predicted true cases")
    plt.xlabel(f'# days since March 4th, 2020')
    plt.legend(loc=4)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.savefig(f'covid_tracking_images/all.png', bbox_inches='tight')
    plt.close()


def plot_state_graph(cases, dates, dates_as_int_array_normalized, days,
                     normalized_cases, popt, state_name):
    plt.title(state_name)
    plt.scatter(dates_as_int_array_normalized, cases, label=f"Confirmed cases")
    plt.scatter(dates_as_int_array_normalized, normalized_cases,
                label=f"Confirmed cases \n(normalized to {round(ASSUMED_RATIO * 100)}% testing positive)")
    plt.plot(days, model(days, *popt), 'g--', label=f"Predicted true cases")
    plt.xlabel(f'# days since {dates.min()}')
    plt.legend(loc=4)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.savefig(f'covid_tracking_images/{state_name}.png', bbox_inches='tight')
    plt.close()


if __name__=="__main__":
    run()

