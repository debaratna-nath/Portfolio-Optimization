import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data

st.title("Portfolio Optimization")


tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]
# st.write(tickers['SYMBOL'])
tickers_ = tickers['SYMBOL']
x = st.multiselect("Select Stocks (NSE Sensex)", tickers_)

# st.write(x[0])
for stock in range(len(x)):
    # st.write(x[stock])
    x[stock] = x[stock] + ".NS"

submit = st.button('Compute')
if submit:

    df = data.DataReader(x, 'yahoo', start='2015/01/01', end='2021/12/31')
    df = df['Adj Close']

    # Log of percentage change
    #because log of percentage change is time additive
    #add 1+x to avoid log(0) error

    #Compute covariance matrix
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    # cov_matrix

    #compute correlation matrix
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    # corr_matrix

    # Yearly returns for individual companies
    ind_er = df.resample('Y').last().pct_change().mean()
    # ind_er

    # Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    # ann_sd

    assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
    assets.columns = ['Returns', 'Volatility']
    # assets
    # df.head()

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(df.columns)
    num_portfolios = 10000
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                        # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(df.columns.tolist()):
        #print(counter, symbol)
        data[symbol+' weight'] = [w[counter] for w in p_weights]
    
    portfolios  = pd.DataFrame(data)
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    # idxmin() gives us the minimum value in the column specified.                               
    st.write("Minimum Volatility Portfolio")
    min_vol_port = pd.DataFrame(min_vol_port).reset_index()
    min_vol_port.columns = ['Index', 'Values']
    st.write(min_vol_port)
    # st.write(df)

    # Finding the optimal portfolio
    rf = 0.01 # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
    st.write("Optimal Risk Portfolio")
    optimal_risky_port = pd.DataFrame(optimal_risky_port).reset_index()
    optimal_risky_port.columns = ['Index', 'Values']
    st.write(optimal_risky_port)