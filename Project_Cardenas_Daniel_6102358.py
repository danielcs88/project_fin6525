# -*- coding: utf-8 -*-
# %% [markdown] tags=[]
"""
# Term Project: FIN6525

by [Daniel Cárdenas [6102358]](https://danielcs88.github.io/)
"""

# %%
from IPython import get_ipython

# This whole line is just to make running the code possible using Google Colab
# and installs depdenencies

if "google.colab" in str(get_ipython()):
    print("Running on Colab")
    get_ipython().run_cell_magic(
        "capture",
        "",
        "! pip install yfinance\n! pip install numpy-financial\n! pip install pandas-bokeh\n! pip3 install pickle5",
    )
    import pickle5 as pickle


else:
    print("Not running on Colab")


# %%
import functools
import operator
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy_financial as npf
import pandas as pd
import pandas_bokeh
import seaborn as sns
import statsmodels.api as sm
import statsmodels.sandbox.stats.runs as runs
import yfinance as yf
from IPython.display import Markdown, display
from matplotlib.dates import date2num
from pandas_datareader import data as pdr
from scipy.stats import gmean

# Plotting parameters to make plots bigger
plt.rcParams["figure.dpi"] = 125
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")

# %%
display(
    Markdown(
        """
   # - [Term Project: Fin6525](#Term-Project:-Fin6525)
   * [Part One: Data](#Part-One:-Data)
     + [A. Table 1](#A.-Table-1)
       - [Table 1: Monthly Returns](#Table-1:-Monthly-Returns)
       - [Table 1: Summary Statistics](#Table-1:-Summary-Statistics)
       - [Annualized Returns](#Annualized-Returns)
     + [B. Table 2: Covariance Matrix](#B.-Table-2:-Covariance-Matrix)
     + [C. Table 3: Correlation Matrix](#C.-Table-3:-Correlation-Matrix)
     + [D. Prospectus Strategy](#D.-Prospectus-Strategy)
   * [Part Two](#Part-Two)
     + [A. CAPM](#A.-CAPM)
     + [B. β](#B.-β)
     + [C. Table 4](#C.-Table-4)
     + [D. Essay: Differences Between Dow Jones And S&P 500](#D.-Essay:-Differences-between-Dow-Jones-and-S&P-500)
     + [E. Runs Test: S&P 500](#E.-Runs-Test:-S&P-500)
       - [Runs Test Interpretation](#Runs-Test-Interpretation)
   * [Part Three](#Part-Three)
       - [A. Table 5](#A.-Table-5)
     + [B. Graph 1](#B.-Graph-1)
       - [Static Graph](#Static-Graph)
       - [Dynamic Graph](#Dynamic-Graph)
   * [Part Four](#Part-Four)
     + [A. Graph 2: Mean Variance Plot](#A.-Graph-2:-Mean-Variance-Plot)
       - [Globally Minimum Variance Portfolio](#Globally-Minimum-Variance-Portfolio)
     + [B. Graph 3: Mean-Variance Frontier](#B.-Graph-3:-Mean-Variance-Frontier)
       - [Random Portfolios](#Random-Portfolios)
         * [Minimum Volatility](#Minimum-Volatility)
         * [Maximum Sharpe Ratio](#Maximum-Sharpe-Ratio)
   * [Part Five: Performance](#Part-Five:-Performance)
     + [Sharpe Measure](#Sharpe-Measure)
     + [Treynor Measure](#Treynor-Measure)
     + [Rankings](#Rankings)
       - [Sharpe Measure](#Sharpe-Measure:-Rankings)
       - [Treynor Measure](#Treynor-Measure:-Rankings)
       - [Geometric Mean](#Geometric-Mean:-Rankings)
"""
    )
)

# %%
# Formatting to display numbers
PERCENT = "{:,.3%}"
CURRENCY = "${:,.2f}"

proper_format = {
    "Arithmetic Mean": PERCENT,
    "Geometric Mean": PERCENT,
    "Standard Deviation": PERCENT,
    "Current Value: $10k": CURRENCY,
}

plt.style.use("seaborn-white")

# %% tags=[]
# pylint: disable=W0105,W0104

# %%
start = datetime(2017, 1, 1)
end = datetime(2022, 1, 1)

# %% [markdown] tags=[]
# ## Part One: Data

# %%
funds = sorted(["COPX", "UNL", "CURE", "TAN", "TECL"])

# %%
tickers = yf.Tickers(funds)

# %%
# Price Data
data = yf.download(funds, start=start, end=end, interval="1mo")[
    "Adj Close"
].dropna()  # Dropping non-trading dates

# %%
data.head()

# %%
# To calculate simple returns we simply call the percent change function
returns = data.pct_change()

# %%
returns.tail()

# %%
returns = returns.dropna()

# %% [markdown]
# ### A. Table 1

# %% [markdown]
# #### Table 1: Monthly Returns

# %%
with pd.option_context("display.float_format", PERCENT.format):
    display(returns)

# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(returns, label=returns.columns)
ax.axvspan(
    date2num(datetime(2020, 2, 1)),
    date2num(datetime(2020, 4, 1)),
    label="COVID-19 Downturn",
    color="grey",
    alpha=0.3,
)
ax.legend(loc=3)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Portfolio Monthly Return")
plt.show()

# %%
returns_m, returns_n = returns.shape

# %% [markdown]
# #### Table 1: Summary Statistics

# %%
table_1 = pd.DataFrame(
    {
        "Arithmetic Mean": returns.mean(),
        "Geometric Mean": (gmean(returns + 1) - 1),
        "Standard Deviation": returns.std(
            ddof=0
        ),  # degrees of freedom=0 for population stats
        "Current Value: $10k": npf.fv(
            rate=(gmean(returns + 1) - 1),
            pmt=0,
            nper=returns_m,
            pv=[-10000] * returns_n,
        ),
    }
)
table_1.style.format(proper_format)

# %% [markdown]
# #### Annualized Returns
#
# And if we decided to look at the annualized returns:

# %% [markdown]
# $$
# APY = (1+r_{\text{month}})^{12} - 1 \\
# \sigma_{\text{yearly}} = \sigma_{\text{month}} \times \sqrt{12}
# $$

# %%
annualized = table_1.copy()
annualized[["Arithmetic Mean", "Geometric Mean"]] = (
    annualized[["Arithmetic Mean", "Geometric Mean"]] + 1
) ** (12) - 1
annualized["Standard Deviation"] = annualized["Standard Deviation"] * (12 ** 0.5)

# %%
annualized.style.format(proper_format)

# %% [markdown]
# ### B. Table 2: Covariance Matrix

# %%
# All formulas related to variances (i.e., standard deviations, covariance)
# will have `dd=0` for the population standard deviation.
returns.cov(ddof=0)

# %% [markdown]
# ### C. Table 3: Correlation Matrix

# %%
returns.corr()

# %%
# sns.set(rc={"figure.figsize": (8, 8)})
sns.heatmap(
    returns.corr(),
    annot=True,
    cmap="RdBu_r",
    square=True,
    vmin=-1,
    vmax=1,
).set_title("Correlation Matrix")
plt.show()

# %% [markdown]
# ### D. Prospectus Strategy
#
# Using each fund’s prospectus or information you find on the web, state in your
# own words the strategy and philosophy of each fund.

# %% [markdown]
# | ETF          | COPX                                                         | CURE                                                         | TAN                                                          | TECL                                                         | UNL                                                          |
# | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
# | **Name**     | Global X Copper Miners ETF                                   | Direxion Daily Healthcare Bull 3X Shares                     | Invesco Solar ETF                                            | Direxion Daily Technology Bull 3x Shares                     | United States 12 Month Natural Gas Fund, LP                  |
# | **About**    | COPX tracks a market-cap-weighted index of global copper mining companies. | CURE provides 3x levered exposure to a market-cap-weighted sub-index of the S&P 500 that tracks the performance of US-listed health care companies. | TAN tracks an index of global solar energy companies selected based on the revenue generated from solar-related business. | TECL provides 3x leveraged exposure to a market-cap weighted index of US large-cap technology companies. | UNL holds the 12 nearest-month NYMEX natural gas futures contracts in equal weights. |
# | **Category** | Basic Materials                                              | Trading--Leveraged Equity                                    | Miscellaneous Sector                                         | Trading--Leveraged Equity                                    | Commodities Focused                                          |
# | **Strategy** | Vanilla                                                      | Vanilla                                                      | Fundamental                                                  | Vanilla                                                      | Laddered                                                     |
# | **Segment**  | Equity: Global Copper Miners                                 | Leveraged Equity: U.S. Health Care                           | Equity: Global Renewable Energy                              | Leveraged Equity: U.S. Information Technology                | Commodities: Energy Natural Gas                              |
# | **Niche**    | ETF                                                          | COPXBroad-based                                              | CURERenewable Energy                                         | TANBroad-based                                               | TECLLaddered                                                 |

# %% [markdown] tags=[]
# ## Part Two

# %% [markdown]
# ### A. CAPM
#
# Using Treasury bill rates and the S&P 500 index, run a “CAPM” regression to
# estimate the beta of each fund. Constant maturity 3-month T-bill rates can be
# obtained on the web site of The Federal Reserve Bank of Saint Louis, Missouri
# (“FRED”). These rates are quoted in an annualized format, so adjust them
# according to your needs.

# %% [markdown]
# Beta is defined as:
#
# $$
# \beta_i = \frac{\sigma_{iM}}{\sigma^{2}_{M}}
# $$

# %%
# Note that Python uses zero-indices
T13W = pdr.DataReader("GS3M", "fred", "02/2017", end)
Rm = yf.download(["^GSPC", "^DJI"], start=start, end=end, interval="1mo")["Adj Close"]
Rm = Rm.pct_change().dropna()

# %% [markdown]
# Since Treasury Bill Rates are quoted annually and already in percentage form,
# we will convert this to decimal and monthly return.
#
#
# According to the [U.S. Treasury](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions),
# the Constant Maturity Rates are _expressed on a simple annualized basis_,
# therefore to convert them to a monthly basis, we will multiply by `12`.

# %%
T13W = T13W.div(100).div(12)

# %% [markdown]
# The traditional equation for the Capital Asset Pricing Model
# (CAPM) is as follows:
#
# $$ R_i = R_f + \beta(R_m -  R_f) $$
#
# yet we are looking to find the our beta coefficient, which we can find by
# re-arranging the equation into:
#
# $$ R_i - R_f = \beta(R_m - R_f) $$
#
# which translates into:
#
# $$ R_\text{Fund} - R_{\text{TBill 13W}} =  \beta(R_{\text{S&P500}} -
# R_{\text{TBill 13W}}) \\
# $$

# %% [markdown]
# ### B. β
#
# Repeat part A using this time the Dow Jones Industrial Average instead of the
# S&P 500 index.

# %%
# Run the regression for each fund with S^P500 as the benchmark
sp500_ols = [sm.OLS(endog=returns[fund], exog=Rm["^GSPC"]).fit() for fund in funds]

# %%
# Run the regression for each fund with the DJIA as the benchmark
djia_ols = [sm.OLS(endog=returns[fund], exog=Rm["^DJI"]).fit() for fund in funds]

# %%
for result in sp500_ols:
    display(result.summary())

# %%
# Extract the beta coefficients
sp500_beta = [sp500_ols[fund].params[0] for fund in range(5)]
djia_beta = [djia_ols[fund].params[0] for fund in range(5)]

# %%
betas = pd.DataFrame([sp500_beta, djia_beta], columns=funds, index=["^GSPC", "^DJI"])
display(Markdown("$\\beta$ per Fund and Market"), betas)

# %%
riskFree = T13W.mean()[0]
returnSP = Rm["^GSPC"].mean()
returnDJIA = Rm["^DJI"].mean()

# %% [markdown]
# $$E(R_i) = R_f + \beta(R_m -  R_f)$$

# %%
E_R = pd.DataFrame()
E_R["^GSPC"] = riskFree + betas.T["^GSPC"] * (returnDJIA - riskFree)
E_R["^DJI"] = riskFree + betas.T["^DJI"] * (returnDJIA - riskFree)

# %%
display(Markdown("$E(R)$ Expected Return explained by $R_m$"))
display(
    E_R.sort_index().T.style.format(PERCENT),
    table_1["Arithmetic Mean"].sort_index().to_frame().T.style.format(PERCENT),
)

# %%
buy_suggestion = pd.DataFrame()
buy_suggestion["^GSPC"] = (
    E_R["^GSPC"].sort_index() < table_1["Arithmetic Mean"].sort_index()
)
buy_suggestion["^DJI"] = (
    E_R["^DJI"].sort_index() < table_1["Arithmetic Mean"].sort_index()
)
display(
    Markdown(
        "Buy Suggestion: if actual return is larger than expected return $R_i > E(R)$"
    )
)
buy_suggestion.sort_index()

# %% [markdown]
# ### C. Table 4
#
# Show the T-bill rates and the two index levels in tabular form.

# %%
table_4 = pd.concat([Rm, T13W], axis=1)
table_4

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(table_4, label=table_4.columns)
ax.axvspan(
    date2num(datetime(2020, 2, 1)),
    date2num(datetime(2020, 4, 1)),
    label="COVID-19 Downturn",
    color="grey",
    alpha=0.3,
)
ax.legend(loc=3)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title(
    "S&P 500, Dow Jones Industrial Average,"
    "\n"
    "Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity"
    "\n"
    "Monthly Returns"
)
plt.show()

# %% [markdown]
# ### D. Essay: Differences between Dow Jones and S&P 500

# %%
display(Markdown("$E(R)$ Expected Return explained by $R_m$"))
display(
    E_R.sort_index().T.style.format(PERCENT),
    table_1["Arithmetic Mean"].sort_index().to_frame().T.style.format(PERCENT),
)

# %%
returns_tidy = pd.melt(returns.reset_index(), id_vars="Date").set_index("Date")
returns_tidy = returns_tidy.rename(columns={"variable": "Fund", "value": "Return"})

# %%
# This complicated code is to make running possible on Google Colab
try:
    tidy_data = pd.concat([pd.concat([table_4] * 5), returns_tidy], axis=1)
    tidy_data.to_pickle("data/tidy_huh.pkl")
except:
    get_ipython().system(
        "wget https://raw.githubusercontent.com/danielcs88/project_fin6525/main/data/tidy_huh.pkl"
    )
    with open("tidy_huh.pkl", "rb") as fh:
        tidy_data = pickle.load(fh)

# %%
sns.set(rc={"figure.figsize": (6, 30)})
sns.lmplot(x="^GSPC", y="Return", col="Fund", data=tidy_data)
plt.show()

# %%
sns.set(rc={"figure.figsize": (6, 30)})
sns.lmplot(x="^DJI", y="Return", col="Fund", data=tidy_data)
plt.show()


# %% [markdown]
# In my particular case, the differences in Expected Return explained by market
# returns by the S&P 500 or the Dow Jones Industrial Average were little.
#
# The only fund that didn’t perform identically was
# [**UNL**](https://finance.yahoo.com/quote/UNL?p=UNL](https://www.google.com/url?q=https://finance.yahoo.com/quote/UNL?p%3DUNL&sa=D&source=editors&ust=1644906676549149&usg=AOvVaw2_AqMFz1d5je1uQkf0XAP0).
# This can be explained by the negative beta values ($\beta_{\text{S&P}}=-0.08$,
# $\beta_{\text{DJIA}}=-0.14$) from its regression with market returns. While this
# fund had the worst mean individual returns ($R_{\text{UNL}}=10.9\%$ monthly), it
# compensated this by having the lowest risk ($\sigma=28.06\%$) of the portfolio.
# UNL also served as a hedge against all other assets in the portfolio and the
# expected return. Its average correlation with all the assets stands at 3.5% (all
# correlations are negative with all assets).
#
# However, by running a correlation analysis between the S&P 500 and the Dow
# Jones, it is easy to see why the results are consistently similar; they are 96%
# correlated with each other.
#
# Another reason, in my opinion why the results are so similar is the ultra-low
# Treasury rates observed throughout the 5-year period. People have little to no
# incentive to purchase Treasury bills.
#
# Lastly, this period has been one for analysis galore. Supply shocks, demand
# shocks. Oil and energy plunging in the heart of the pandemic and now reaching
# almost $100 a barrel.
#
# It has been characterized by a very easy monetary policy throughout, it was
# defined by three key events.
#
# 1. Trump Administration monetary policy:
#
#    1. Tax Cuts and Jobs Act (TCJA)
#
# 2. COVID-19 Market Crash
#    1. Coronavirus Aid, Relief, and Economic Security Act (CARES Act): $2.2
#       trillion economic stimulus
#    2. Low Treasury Yields
#    3. Federal Reserve
#       1. Monetary Easing and Purchase of Treasury Bonds
# 3. COVID-19 Recovery
#    1. Demand shocks
#    2. Supply chain inefficiencies
#
# People have never had more reasons to trade stocks. All between being
# quarantined in a pandemic with little to do and federal money being put in
# Americans’ pockets, has produced sky-high records in the market.
#

# %%
pd.melt(pd.concat([returns, table_4], axis=1))

# %%
sns.displot(
    (
        pd.melt(
            pd.concat([returns, table_4], axis=1)[
                ["COPX", "CURE", "TAN", "TECL", "UNL", "^DJI", "^GSPC"]
            ]
        )
    ),
    x="value",
    hue="variable",
    kind="kde",
)

# %%
sns.set(rc={"figure.figsize": (8, 8)})
sns.heatmap(
    pd.concat([returns, table_4], axis=1).corr(),
    annot=True,
    cmap="RdBu_r",
    square=True,
    vmin=-1,
    vmax=1,
).set_title("Correlation Matrix: Funds and (S&P500, DJIA, Treasury)")
plt.show()

# %%
sns.set(rc={"figure.figsize": (6, 6)})
sns.heatmap(
    table_4.corr(),
    annot=True,
    # cmap="RdBu_r",
    square=True,
    vmin=-1,
    vmax=1,
).set_title("Correlation Matrix: Table 4 (S&P500, DJIA, Treasury)")
plt.show()

# %%
# sns.reset_defaults()

# %% [markdown]
# ### E. Runs Test: S&P 500

# %% [markdown]
# $$
# \begin{align}
# Z &= \frac{R-\bar{x}}{\sigma} \\
# \text{where } R &= \text{number of runs;} \\
# \bar{x} &= \frac{2n_1 n_2}{n_1+ n_2} + 1 \\
# \sigma^2 &= \frac{2n_1 n_2 (2n_1 n_2 - n_1 - n_2)}{(n_1+n_2)^2 (n_1 + n_2 - 1)} \\
# n_1, n_2 &= \text{number of observations in each category} \\
# Z &= \text{standard normal variable}
# \end{align}
# $$

# %%
sp500 = Rm["^GSPC"]


# %%
def num_runs(array) -> int:
    """
    Performs a Runs Test on an array of values, to calculate the number of runs
    and location (indices of runs)

    Parameters
    ----------
    array : np.ndarray
        [description]

    Returns
    -------
    int
        Number of runs in an array.
    """

    # Check where the indices change
    # array[:-1] = all elements except last
    # array[1:] =  all elments except first
    # np.sign will check sign of elements in array
    # - convert to 1 if positive
    # - convert to -1 if negative
    array = np.array(array)
    indices = np.where(np.sign(array[:-1]) != np.sign(array[1:]))[0] + 1
    # return len(indices), indices
    return len(indices)


# %%
Z = num_runs(sp500)

# %%
np.sign(sp500).value_counts()


# %%
def runs_test(array) -> float:
    R = num_runs(array)
    n1, n2 = list(np.sign(array).value_counts().values)
    x_bar = (2 * n1 * n2) / (n1 + n2) + 1
    sigma_sq = (
        2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    )
    Z = (R - x_bar) / np.sqrt(sigma_sq)
    return Z


# %%
Z = runs_test(sp500)
Z

# %%
Z, p = runs.runstest_1samp(Rm["^GSPC"])
runs.runstest_1samp(Rm["^GSPC"])

# %% [markdown]
# #### Runs Test Interpretation

# %%
np.sign(sp500).value_counts()

# %%
display(
    Markdown(
        f"""Our Z statistic of **{Z:.3f}** is not close to the standard
normal distribution mean of 0.

We cannot be 95 percent certain that our observed stock prices
did not happen by chance unless we get a Z statistic whose absolute value is
1.96 or greater.

I did all the work above _show my work_, but thankfully Python does have a
faster method for this through `statsmodels`."""
    )
)

# %%
display(
    Markdown(
        f"""The z-test statistic turns out to be **$Z={Z:.3f}$** and the
corresponding p-value is **$p={p:.3f}$**. Since this p-value is not less than
α = .05, we fail to reject the null hypothesis. We have sufficient evidence to
say that the data was produced in a random manner."""
    )
)

# %% [markdown] tags=[]
# ## Part Three

# %% [markdown]
# #### A. Table 5

# %% [markdown]
# A. Construct an equally-weighted portfolio of your five funds. Prepare a table
# showing the arithmetic mean return, geometric mean return, and standard
# deviation of return for the five-fund portfolio over the five years [TABLE 5].

# %% [markdown]
# To calculate the standard deviation of the portfolio, we will use the formula
# of the variance of the portfolio and take its square root.
#
# $$
# \Large
# \begin{gather} \notag
# \begin{aligned}
# \sigma_p &= \sqrt{w' V w} \\
# &= \sqrt{\begin{bmatrix} 0.2 & 0.2 & 0.2 & 0.2 & 0.2 \end{bmatrix} \mathbf{V}
# \begin{bmatrix} 0.2 \\ 0.2 \\ 0.2 \\ 0.2 \\ 0.2 \end{bmatrix}}
# \end{aligned}
# \end{gather}
# $$

# %%
# Equal weighted portfolio weights
w = np.array([[0.2] * 5])

# %%
table_5 = pd.Series(
    {
        "Arithmetic Mean": returns.mean().dot(w.T).item(0),
        "Geometric Mean": (gmean(returns + 1) - 1).dot(w.T).item(0),
        "Standard Deviation": np.sqrt(w.dot(returns.cov(ddof=0)).dot(w.T).item(0)),
    }
).to_frame()
table_5 = table_5.T
table_5.index = ["Equally Weighted Portfolio"]
table_5.style.format(proper_format)

# %% [markdown]
# ### B. Graph 1
#
# Using Excel, prepare a graph showing the five-year performance of each of your
# funds and the five-fund portfolio. This chart should show the dollar value of
# an initial \\$10,000 investment evolving month-by-month over the five-year
# period.

# %% [markdown]
# #### Static Graph

# %%
portfolio_five = pd.DataFrame(
    {"Equally Weigthed Portfolio": 10000 * (1 + returns.mean(axis=1)).cumprod()}
)

# %%
sep_funds = (1 + returns).cumprod() * 10000

# %%
investments = pd.concat([portfolio_five, sep_funds], axis=1)

# %%
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(
    investments["Equally Weigthed Portfolio"],
    label="Equally Weighted Portfolio",
    linestyle=":",
)
ax.plot(investments[funds], label=funds)
ax.axvspan(
    date2num(datetime(2020, 2, 1)),
    date2num(datetime(2020, 4, 1)),
    label="COVID-19 Downturn",
    color="grey",
    alpha=0.3,
)
ax.legend()

ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.set_ylabel("Terminal Value")
ax.set_title("Graph 1: Portfolio Monthly Return vs. Individual Funds")
ax.set_ylim(1500, 145000)
plt.show()

# %%
pandas_bokeh.output_notebook()

# %% [markdown]
# #### Dynamic Graph
#
# If I wanted a more interactive plot, I could use `pandas-bokeh`

# %%
investments.plot_bokeh.line(
    title="Graph 1: Individual Funds vs Portfolio",
    disable_scientific_axes="y",
    number_format="‘$0,0.0’",
    ylabel="Terminal Value [$]",
    legend="top_left",
    figsize=(1024, 600),
    panning=False,
    zooming=False,
)

# %% [markdown] tags=[]
# ## Part Four

# %% [markdown]
# Using the five-year performance statistics of your five funds and the
# five-fund portfolio, determine and show graphically the efficient set using
# the following:

# %% [markdown]
# ### A. Graph 2: Mean Variance Plot
#
# This is merely a standard deviation / expected return plot showing six points,
# one for each fund and one for the five-fund portfolio. Identify the point that
# shows the best return per unit of risk.

# %%
mean_std = [table_1.columns[i] for i in [0, 2]]
mean_var = pd.concat([table_1[mean_std], table_5[mean_std]])
mean_var.style.format(proper_format)

# %%
mean_var = mean_var.reset_index()
mean_var.columns = ["Index", "Expected Return", "Standard Deviation"]

# %%
mean_var_plot = sns.scatterplot(
    data=mean_var, x="Standard Deviation", y="Expected Return", hue="Index"
).set(title="Mean Variance Plot")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))


# %% [markdown]
# #### Globally Minimum Variance Portfolio
#
# $$
# \LARGE
# \begin{matrix}
# \notag
# \mu_{*} = \frac{ \mathbf{1}' \mathbf{V}^{-1} \mathbf{\mu} } {\mathbf{1}' \mathbf{V}^{-1} \mathbf{1}  } &
# \sigma_{*}^{2} = (\mathbf{1}' \mathbf{V}^{-1} \mathbf{1})^{-1} &
# w_{*} = \frac{\mathbf{V}^{-1} \mathbf{1}}{\mathbf{1}' \mathbf{V}^{-1} \mathbf{1}}
# \end{matrix}
# $$

# %%
mu = table_1["Arithmetic Mean"]
V = returns.cov(ddof=0)
one = np.ones((5, 1))

# %%
w_star = pd.DataFrame(
    np.linalg.inv(V).dot(one) / (one.T.dot(np.linalg.inv(V))).dot(one),
    index=returns.columns,
    columns=["w*"],
)
w_star

# %%
mu_star = (
    one.T.dot(np.linalg.inv(V)).dot(mu) / (one.T.dot(np.linalg.inv(V))).dot(one).item(0)
).item(0)
print("μ* =", f"{mu_star:,.3%} monthly")
print("μ* =", f"{((1+mu_star)**12)-1:,.3%} monthly")

# %%
var_star = np.linalg.inv((one.T.dot(np.linalg.inv(V))).dot(one)).item(0)
print("σ²* =", f"{var_star:.4f}")
print("σ* =", f"{np.sqrt(var_star):,.4%}")

# %% [markdown]
# As reference let's add these points to our Mean Variance Plot

# %%
mean_var = mean_var.append(
    pd.Series(
        {
            "Index": "Global Minimum Variance Portfolio",
            "Expected Return": mu_star,
            "Standard Deviation": np.sqrt(var_star),
        }
    ),
    ignore_index=True,
)

# %%
with pd.option_context("display.float_format", PERCENT.format):
    display(mean_var)

# %%
mean_var_plot = sns.scatterplot(
    data=mean_var, x="Standard Deviation", y="Expected Return", hue="Index"
).set(title="Mean Variance Plot")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))


# %% [markdown]
# ### B. Graph 3: Mean-Variance Frontier
#
# Using the matrix Excel-based techniques learned in class, use the five funds
# and their statistics to derive the mean-variance frontier. Draw a plot showing
# the frontier as well as the 5 mean-variance points of the five individual
# funds.

# %% [markdown]
# **Efficient Portfolio Frontier**
# $$
# \Large
# \sigma^{2}_{p} = \begin{bmatrix}E(R_p) \\ 1 \end{bmatrix}'  \left(\begin{bmatrix}\mu_i & 1 \\ \vdots & \vdots \\ \mu_n & 1\end{bmatrix}' \mathbf{V}^{-1} \begin{bmatrix}\mu_i & 1 \\ \vdots & \vdots \\ \mu_n & 1\end{bmatrix}\right)^{-1} \begin{bmatrix}E(R_p) \\ 1\end{bmatrix}
# $$
#
# **Optimal Weights**
# $$
# \Large
# \underbrace{w}_{n \times 1} = \underbrace{\underbrace{\underbrace{\mathbf{V}^{-1}}_{n \times n} \cdot \underbrace{\begin{bmatrix}\mu_i & 1 \\ \vdots & \vdots \\ \mu_n & 1\end{bmatrix}}_{n \times 2} \cdot }_{n \times 2}\underbrace{\left(\underbrace{\begin{bmatrix}\mu_i & 1 \\ \vdots & \vdots \\ \mu_n & 1\end{bmatrix}'}_{2 \times n} \cdot \underbrace{\mathbf{V}^{-1}}_{n \times n} \cdot \underbrace{\begin{bmatrix}\mu_i & 1 \\ \vdots & \vdots \\ \mu_n & 1\end{bmatrix}}_{n \times 2}\right)^{-1}}_{2 \times 2} \cdot \underbrace{\begin{bmatrix}E(R_p) \\ 1\end{bmatrix}}_{2 \times 1}}_{n \times 1}
# $$

# %% [markdown]
# I will convert all returns into annual returns to better interpret and
# visualize results:

# %%
def mean_variance_frontier(
    required_return: float, timeframe: str = "Y", df: pd.DataFrame = returns
) -> List:
    """
    Helper function to calculate the mean-variance frontier.

    Parameters
    ----------
    required_return : float
        Required rate of return.
    timeframe : str, optional
        Annual timeframe for returns, if different parameter will return monthly
        results, by default "Y"
    df : pd.DataFrame, optional
        DataFrame to pass to analyze, by default returns

    Returns
    -------
    List
        Returns a list of the following elements:
        [Expected Return, Standard Deviation, W1, W2, W3, W4, W5]
    """

    if timeframe == "Y":
        mu = ((1 + returns.mean()) ** 12) - 1
        V_inv = np.linalg.inv(df.cov(ddof=0) * 12)
    else:
        V_inv = np.linalg.inv(df.cov(ddof=0))
        mu = df.mean()

    ones = np.ones(len(df.columns))
    mu_one = np.column_stack((mu, ones))

    factor_1 = V_inv.dot(mu_one)
    factor_2 = np.linalg.inv((mu_one.T.dot(V_inv)).dot(mu_one))
    required_vector = np.array([[required_return], [1]])

    w = np.squeeze((factor_1.dot(factor_2)).dot(required_vector))
    expected_return = np.squeeze(w.T.dot(mu)).item(0)
    sigma = np.squeeze(
        np.sqrt((required_vector.T.dot(factor_2)).dot(required_vector))
    ).item(0)

    mean_var = pd.DataFrame((expected_return, sigma)).T
    mean_var.columns = ["E(Rp)", "SD(p)"]

    return functools.reduce(operator.iconcat, [[expected_return, sigma], w], [])


# %%
def frontier_df(required_return: float) -> pd.DataFrame:
    """
    Helper function to generate a dataframe of the mean-variance frontier.

    Parameters
    ----------
    required_return : float
        Required rate of return.

    Returns
    -------
    pd.DataFrame
        Mean Variance Frontier in DataFrame format.
    """
    frontier = [mean_variance_frontier(i) for i in np.linspace(0, required_return, 500)]
    frontier_pd = pd.DataFrame(frontier)
    frontier_pd.columns = functools.reduce(
        operator.iconcat, [["E(Rp)", "SD(P)"], returns.columns], []
    )

    return frontier_pd


# %%
frontier = frontier_df(0.4)

with pd.option_context("display.float_format", PERCENT.format):
    display(frontier)


# %% [markdown]
# Now to annualize the results:

# %%
def monthly_to_annual(df: pd.DataFrame = mean_var) -> pd.DataFrame:
    """
    Helper function to convert monthly returns to annual returns.

    Parameters
    ----------
    df : pd.DataFrame, optional
        _description_, by default mean_var

    Returns
    -------
    pd.DataFrame
        Annualized returns in DataFrame format.
    """

    df["Expected Return"] = ((df["Expected Return"] + 1) ** 12) - 1
    df["Standard Deviation"] = df["Standard Deviation"] * (12 ** 0.5)

    return df


# %%
annual_mean_var = monthly_to_annual()

# %%
with pd.option_context("display.float_format", PERCENT.format):
    display(annual_mean_var)

# %%
sns.set(rc={"figure.figsize": (10, 8)})
mean_var_plot = sns.scatterplot(data=frontier_df(0.4), x="SD(P)", y="E(Rp)", s=5).set(
    title="Mean Variance Plot"
)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# Since the annual return for one of my funds is nearly 100%, the shape of this
# frontier compared to each fund looks strange.

# %%
sns.scatterplot(
    data=annual_mean_var,
    x="Standard Deviation",
    y="Expected Return",
    hue="Index",
    style="Index",
).set(title="Mean Variance Plot")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

mean_var_plot = sns.scatterplot(data=frontier, x="SD(P)", y="E(Rp)").set(
    title="Mean Variance Plot"
)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# #### Random Portfolios

# %% [markdown]
# We can also generate random portfolios by the following:

# %%
port_returns = []
port_sd = []
port_w = []

num_assets = len(funds)
num_portfolios = 25000

for _ in range(num_portfolios):
    # Random generation of weights
    weights = np.random.random(num_assets)
    # Normalization of weights
    weights /= np.sum(weights)
    port_w.append(weights)

    port_returns.append(np.dot(weights, ((1 + returns.mean()) ** 12) - 1))
    port_var = (
        12 * returns.cov(ddof=0).mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    )
    sd_port = np.sqrt(port_var)
    port_sd.append(sd_port)

# %%
data = {"Expected Return": port_returns, "Standard Deviation": port_sd}

for counter, symbol in enumerate(returns.columns.tolist()):
    data[symbol] = [w[counter] for w in port_w]

# %%
randomPortfolios = pd.DataFrame(data)

# %%
randomPortfolios.head()

# %%
mean_var_plot = sns.scatterplot(
    data=randomPortfolios,
    x="Standard Deviation",
    y="Expected Return",
).set(title="Mean Variance Plot")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# ##### Minimum Volatility

# %%
# idxmin() gives us the min value per column specified
min_vol_port = randomPortfolios.iloc[randomPortfolios["Standard Deviation"].idxmin()]
min_vol_port = min_vol_port.to_frame().T
min_vol_port.index = ["Random Minimum Variance Portfolio"]

# %%
min_vol_port

# %% [markdown]
# Interesting, this minimum random portfolio beat our minimum variance portfolio
# in terms of expected return but not in minimizing volatility. Yet, it doesn't
# incur in short-selling.

# %%
display(annual_mean_var, w_star)

# %% [markdown] tags=[]
#  ##### Maximum Sharpe Ratio

# %%
annual_RiskFree = riskFree * 12
optimal_risky_port = randomPortfolios.iloc[
    (
        (randomPortfolios["Expected Return"] - annual_RiskFree)
        / randomPortfolios["Standard Deviation"]
    ).idxmax()
]

optimal_risky_port = optimal_risky_port.to_frame().T
optimal_risky_port.index = ["Random Maximum Sharpe Ratio Portfolio"]

# %%
optimal_risky_port

# %%
annual_mean_var = pd.concat(
    [
        annual_mean_var.set_index("Index"),
        min_vol_port[["Expected Return", "Standard Deviation"]],
        optimal_risky_port[["Expected Return", "Standard Deviation"]],
    ]
).reset_index()

# %%
annual_mean_var.columns = ["Index", "Expected Return", "Standard Deviation"]

# %%
with pd.option_context("display.float_format", PERCENT.format):
    display(annual_mean_var)

# %%
randomPortfolios

# %%
mean_var_plot = sns.scatterplot(
    data=randomPortfolios, x="Standard Deviation", y="Expected Return", s=10
).set(title="Mean Variance Plot")


sns.scatterplot(data=frontier_df(0.8), x="SD(P)", y="E(Rp)", s=15).set(
    title="Mean Variance Plot"
)

results_plot = sns.scatterplot(
    data=annual_mean_var, x="Standard Deviation", y="Expected Return", hue="Index", s=80
).set(title="Mean Variance Plot")

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# After looking at this graph we can see that while the randomly generated
# Maximum Sharpe Ratio Portfolio lies outside the frontier, it is likely (at
# least I think so) that this is due to the lack of constraint of short-seling
# in the mean variance frontier matrix formulation

# %% [markdown] tags=[]
# ## Part Five: Performance
#
# Rank the performance of the five funds and the five-fund portfolio according
# to the following criteria:
#
# - The Sharpe Measure
# - The Treynor Measure
# - Geometric Mean Return

# %% [markdown]
# ### Sharpe Measure
#
# > Note: The textbook claims the following:
#
#
# ![image](https://user-images.githubusercontent.com/13838845/153757379-29865a3b-f999-44b5-b9bd-4e6b648a2e2f.png)
#
# $$
# \text{Sharpe Measure} = \frac{\overline{R} - R_f}{\sigma}
# $$

# %%
with pd.option_context("display.float_format", PERCENT.format):
    display(annual_mean_var.set_index("Index"))

# %%
annual_mean_var["Sharpe Ratio"] = (
    annual_mean_var["Expected Return"] - annual_RiskFree
) / annual_mean_var["Standard Deviation"]

# %%
randomPortfolios["Sharpe Ratio"] = (
    randomPortfolios["Expected Return"] - riskFree
) / randomPortfolios["Standard Deviation"]

# %%
randomPortfolios

# %%
sns.set(rc={"figure.figsize": (16, 9)})

sns.scatterplot(data=frontier_df(0.8), x="SD(P)", y="E(Rp)", s=30)

results_plot = sns.scatterplot(
    data=annual_mean_var,
    x="Standard Deviation",
    y="Expected Return",
    style="Index",
    hue="Index",
    s=600,
    marker="*",
    palette="bright",
)

mean_var_plot = sns.scatterplot(
    data=randomPortfolios,
    x="Standard Deviation",
    y="Expected Return",
    palette="viridis",
    hue="Sharpe Ratio",
    s=15,
).set(title="Mean Variance Plot: Random Portfolios by Sharpe Ratios")

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# ### Treynor Measure

# %% [markdown]
# $$
# \text{Treynor Measure} = \frac{\overline{R} - R_f}{\beta}
# $$

# %%
all_betas = list(list(betas.mean()))

# %%
all_betas.append(
    np.array([0.2] * 5).dot(list(betas.mean()))
)  # equally weighted portfolio

# %%
all_betas.append(w_star.T.dot(betas.mean())[0])

# %%
all_betas.append(min_vol_port[funds].dot(list(betas.mean()))[0])
all_betas.append(optimal_risky_port[funds].dot(list(betas.mean()))[0])

# %%
annual_mean_var["Beta"] = all_betas

# %%
annual_mean_var["Treynor Ratio"] = annual_mean_var["Sharpe Ratio"] = (
    annual_mean_var["Expected Return"] - annual_RiskFree
) / annual_mean_var["Beta"]

# %%
annual_mean_var.set_index("Index").style.format(
    {"Expected Return": PERCENT, "Standard Deviation": PERCENT}
)

# %%
annual_geo = list(((table_1["Geometric Mean"] + 1) ** 12) - 1)

# %%
all_geo = list(((table_1["Geometric Mean"] + 1) ** 12) - 1)

# %%
all_geo.append(np.array([0.2] * 5).dot(annual_geo))  # equally weighted portfolio

# %%
all_geo.append(w_star.T.dot(annual_geo)[0])

# %%
all_geo.append(min_vol_port[funds].dot(annual_geo)[0])
all_geo.append(optimal_risky_port[funds].dot(annual_geo)[0])

# %%
annual_mean_var["Geometric Mean"] = all_geo

# %% [markdown]
# ### Rankings

# %% [markdown]
# #### Sharpe Measure: Rankings

# %%
annual_mean_var.set_index("Index").sort_values(
    by="Sharpe Ratio", ascending=False
).style.format(
    {
        "Expected Return": PERCENT,
        "Standard Deviation": PERCENT,
        "Geometric Mean": PERCENT,
    }
)

# %%
annual_mean_var.set_index("Index")["Sharpe Ratio"].plot(
    kind="barh", title="Sharpe Ratios"
)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# #### Treynor Measure: Rankings

# %%
annual_mean_var.set_index("Index").sort_values(
    by="Treynor Ratio", ascending=False
).style.format(
    {
        "Expected Return": PERCENT,
        "Standard Deviation": PERCENT,
        "Geometric Mean": PERCENT,
    }
)

# %%
annual_mean_var.set_index("Index")["Treynor Ratio"].plot(
    kind="barh", title="Treynor Ratio"
)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

# %% [markdown]
# #### Geometric Mean: Rankings

# %%
annual_mean_var.set_index("Index").sort_values(
    by="Geometric Mean", ascending=False
).style.format(
    {
        "Expected Return": PERCENT,
        "Standard Deviation": PERCENT,
        "Geometric Mean": PERCENT,
    }
)

# %%
annual_mean_var.set_index("Index")["Geometric Mean"].plot(
    kind="barh", title="Geometric Mean"
)
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
