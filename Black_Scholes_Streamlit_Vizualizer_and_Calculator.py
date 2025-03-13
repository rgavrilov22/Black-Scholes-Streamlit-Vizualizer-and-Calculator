import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from datetime import date,timedelta
import yfinance as yf
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>European Option Pricing using Black-Scholes-Merton Model</h1>
    """, unsafe_allow_html=True)

class blackscholesmodel:
    def __init__(self,S,X,t,r,sigma):
        self.S = S
        self.X = X
        self.t = t
        self.r = r
        self.sigma = sigma
    def d1(self):
        return (np.log(self.S / self.X) + (self.r + 0.5 * (self.sigma ** 2)) * self.t) / (self.sigma * np.sqrt(self.t))
    def d2(self):
        return self.d1() - (self.sigma * np.sqrt(self.t))
    def call_price(self):
        return (self.S * norm.cdf(self.d1(), 0, 1) - self.X * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1))
    def put_price(self):
        return (self.X * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1) - self.S * norm.cdf(-self.d1(), 0, 1))

def generate_heatmap(S, X, t, r, sigma_range, stock_price_range, option_type):
    sigma_values = np.linspace(*sigma_range, 10)
    stock_prices = np.linspace(*stock_price_range, 10)

    price_matrix = np.zeros((len(stock_prices), len(sigma_values)))

    for i, stock_price_element in enumerate(stock_prices):
        for j, sigma_element in enumerate(sigma_values):
            bs_model = blackscholesmodel(stock_price_element, X, t, r, sigma_element)
            if option_type == 'call':
                price_matrix[i, j] = bs_model.call_price()
            elif option_type == 'put':
                price_matrix[i, j] = bs_model.put_price()

    plt.figure(figsize=(10, 8))
    sns.heatmap(price_matrix, xticklabels=np.round(stock_prices, 2), yticklabels=np.round(sigma_values, 2),
                cmap='RdYlGn', annot=True, fmt='.3g')
    plt.xlabel('Stock Price')
    plt.ylabel('Volatility')
    plt.title(f'{option_type.capitalize()} Option Price Heatmap')
    st.pyplot(plt.gcf())

class greeks(blackscholesmodel):
    def delta_call(self):
        return norm.cdf(self.d1(),0,1)
    def delta_put(self):
        return -norm.cdf(-(self.d1()),0,1)
    def gamma(self):
        return (norm.pdf(self.d1(),0,1)) / (self.S * self.sigma * np.sqrt(self.t))
    def theta_call(self):
        return (-self.S*norm.pdf(self.d1(), 0, 1)*self.sigma/(2*np.sqrt(self.t)) - self.r*self.X*np.exp(-self.r*self.t)*norm.cdf(self.d2(), 0, 1))/365
    def theta_put(self):
        return (-self.S*norm.pdf(self.d1(), 0, 1)*self.sigma/(2*np.sqrt(self.t)) + self.r*self.X*np.exp(-self.r*self.t)*norm.cdf(-self.d2(), 0, 1))/365
    def vega(self):
        return (self.S * norm.pdf(self.d1(), 0, 1) * np.sqrt(self.t))/100
    def rho_call(self):
        return (self.X * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)) / 100
    def rho_put(self):
        return (-self.X * self.t * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1))/100

def greeks_df(S, X, t, r, sigma):
    get_greeks = greeks(S, X, t, r, sigma)
    dat = {
        '':['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        'Call': [
            get_greeks.delta_call(),
            get_greeks.gamma(),
            get_greeks.vega(),
            get_greeks.theta_call(),
            get_greeks.rho_call(),
        ],
        'Put': [
            get_greeks.delta_put(),
            get_greeks.gamma(),
            get_greeks.vega(),
            get_greeks.theta_put(),
            get_greeks.rho_put(),
        ],
    }
    df = pd.DataFrame(dat,index = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
    df.reset_index()
    df.rename(columns={"index": "Greek"},inplace=True)
    return df

def find_greeks(option_type, greek, S, X, t, r, sigma):
    get_greeks = greeks(S, X, t, r, sigma) 
    if greek == "delta":
        return get_greeks.delta_call() if option_type == 'Call' else get_greeks.delta_put()
    elif greek == "gamma":
        return get_greeks.gamma()
    elif greek == "theta":
        return get_greeks.theta_call() if option_type == 'Call' else get_greeks.theta_put()
    elif greek == "vega":
        return get_greeks.vega()
    elif greek == "rho":
        return get_greeks.rho_call() if option_type == 'Call' else get_greeks.rho_put()

def plot_greeks(riskfree, Spot, eXercise, time, stdev, type, greek):
    fig = go.Figure()
    line_color = 'grey' if greek in ['gamma', 'vega'] else ('green' if type == 'Call' else 'red')
    min_s, max_s = Spot * 0.91, Spot * 1.09
    stock_prices = np.linspace(min_s, max_s, 200)
    greek_values = [find_greeks(type, greek, s, eXercise, time, riskfree, stdev) for s in stock_prices]
    current_greek_value = find_greeks(type, greek, Spot, eXercise, time, riskfree, stdev)
    
    fig.add_trace(go.Scatter(x=stock_prices, y=greek_values, mode='lines', name=f'{greek.capitalize()}', line=dict(color=line_color, width=3)))
    fig.add_trace(go.Scatter(x=[Spot], y=[current_greek_value], mode='markers', name=f'Current {greek.capitalize()}', marker=dict(color='white', size=9)))
    
    fig.update_layout(title=f'{greek.capitalize()} vs Spot Price ({type})',
                      xaxis_title='Spot Price',
                      yaxis_title=greek.capitalize())
    return fig

def main():
    #sidebar (model inputs)
    st.sidebar.header("Model Inputs")
    stock_input = st.sidebar.text_input("Ticker","SPY")
    stock_hist = yf.Ticker(stock_input).history()
    current_stock_price = round(stock_hist['Close'].iloc[-1],2)
    treasury_ticker = yf.Ticker("^TNX")
    current_treasury_yield = round(treasury_ticker.history(period='1d')['Close'].iloc[-1],2)
    S = st.sidebar.number_input('Stock Price (default is SPX)', value = current_stock_price)
    X = st.sidebar.number_input('Exercise Price', value = current_stock_price *1.05)
    t = (-(date.today() - st.sidebar.date_input('Expiration Date', date.today()+timedelta(days=1),min_value=date.today(),format="MM/DD/YYYY"))).days/365
    r = (st.sidebar.number_input('Risk-free rate (%)', value = current_treasury_yield))/100
    sigma = (st.sidebar.number_input('Annual Volatility (in %)', value = 20))/100
    st.sidebar.divider()
    #sidebar (heatmap parameters)
    st.sidebar.header("Heatmap Parameters")
    sigma_range_min = (st.sidebar.slider("Minimum Volatility",0.01,100.00,48.00))/100
    sigma_range_max = (st.sidebar.slider("Maximum Volatility",0.01,100.00,52.00))/100
    sigma_range = (sigma_range_min,sigma_range_max)
    S_range_min = st.sidebar.number_input("Minimum Underlying Price",value = S*0.99)
    S_range_max = st.sidebar.number_input("Maximum Underlying Price",value = S*1.01)
    S_range = (S_range_min,S_range_max)
    #sidebar (acknowledgements)
    st.sidebar.divider()
    st.sidebar.text("Created by:")
    st.sidebar.page_link("https://www.linkedin.com/in/roman-m-gavrilov/", label = 'Roman Gavrilov',icon=":material/person:")
    
    #get base call and put price
    price_call = blackscholesmodel(S,X,t,r,sigma).call_price()
    price_put = blackscholesmodel(S,X,t,r,sigma).put_price()

    #Print call and put price
    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        with stylable_container(
            key="call_cont",
            css_styles=["""
            {
                text-align: center;
                background-color: green;
                border-radius: 1em;
                padding: 0.5em;
            }
            """]
        ):

            st.metric(label='CALL VALUE', value=f'${round(price_call, 2)}')

    with col2:
        with stylable_container(
                key="put_cont",
                css_styles=["""
                {
                    text-align: center;
                    background-color: red;
                    border-radius: 1em;
                    padding: 0.5em;
                }
                """]
        ):
            st.metric(label = "PUT VALUE", value = f'${round(price_put,2)}')

    #Greeks dataframe
    st.divider()
    dat_container = st.container(border=True)
    dat_container.subheader('The Greeks', divider ='grey')
    greeks_dat = greeks_df(S,X,t,r,sigma)
    dat_container.dataframe(greeks_dat,hide_index=True,use_container_width=True)

    #Heatmap
    st.divider()
    heatmap_container = st.container(border=True)
    heatmap_container.subheader('Option Price Heatmap',divider='grey')
    col3, col4 = heatmap_container.columns(2)
    with col3:
        generate_heatmap(S, X, t, r, sigma_range, S_range, option_type='call')
    with col4:
        generate_heatmap(S, X, t, r, sigma_range, S_range, option_type='put')
    st.divider()

    #Graphing greeks
    st.header('Graphs of the Greeks')
    cols = st.columns(2)
    greek_types = ['delta', 'gamma', 'vega', 'theta','rho']
    option_types = ['Call', 'Put']

    graphs = []
    for greek in greek_types:
        if greek in ['gamma', 'vega']:
            graphs.append(plot_greeks(r, S, X, t, sigma, "Call", greek))
        else:
            for option_type in option_types:
                graphs.append(plot_greeks(r, S, X, t, sigma, option_type, greek))

    for i, fig in enumerate(graphs):
        with cols[i % 2]:
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()