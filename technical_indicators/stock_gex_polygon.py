import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm

from config import POLYGON_API_KEY

def calculate_gamma(S, K, T, r, sigma, option_type):
    """
    Calculate gamma for an option using the Black-Scholes formula
    S: spot price
    K: strike price
    T: time to expiration in years
    r: risk-free rate
    sigma: volatility
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def fetch_stock_price(ticker, api_key):
    """Fetch current stock price from Polygon.io"""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    params = {'apiKey': api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data['ticker']['lastTrade']['p']




def fetch_options_data(ticker, date, contract_type, api_key, limit=250):
    """Fetch options data from Polygon.io"""
    base_url = 'https://api.polygon.io/v3/snapshot/options'
    url = f"{base_url}/{ticker}"
    params = {
        'strike_price.gte': 0,
        'expiration_date.gte': date.strftime('%Y-%m-%d'),
        'contract_type': contract_type,
        'limit': limit,
        'sort': 'expiration_date',
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def calculate_gex(ticker, api_key, risk_free_rate=0.05):
    """
    Calculate GEX (Gamma Exposure) for a given stock
    GEX(T,S) = (COI(T,S) - POI(T,S)) * Gamma(T,S) * Spot price * 100
    """
    # Get current stock price
    try:
        spot_price = fetch_stock_price(ticker, api_key)
    except Exception as e:
        raise ValueError(f"Error fetching stock price: {str(e)}")
    current_date = datetime.now().date()

    # Initialize results storage
    gex_data = []

    # Fetch both puts and calls
    try:
        put_data = fetch_options_data(ticker, current_date, 'put', api_key)
    except Exception as e:
        raise ValueError(f"Error fetching put options data: {str(e)}")
    try:
        call_data = fetch_options_data(ticker, current_date, 'call', api_key)
    except Exception as e:
        raise ValueError(f"Error fetching call options data: {str(e)}")

    # Process all expiration dates
    if 'results' in put_data and 'results' in call_data:
        # Group options by expiration date
        expirations = set()
        for opt in put_data['results'] + call_data['results']:
            expirations.add(opt['expiration_date'])

        for exp_date in sorted(expirations):
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d').date()
            T = (exp_datetime - current_date).days / 365.0  # Time to expiration in years

            # Skip if expired
            if T <= 0:
                continue

            # Calculate GEX for each strike price
            strikes = set()
            for opt in put_data['results'] + call_data['results']:
                if opt['expiration_date'] == exp_date:
                    strikes.add(opt['strike_price'])

            for strike in sorted(strikes):
                # Get open interest for this strike and expiration
                put_oi = sum(opt.get('open_interest', 0)
                             for opt in put_data['results']
                             if opt['expiration_date'] == exp_date and opt['strike_price'] == strike)

                call_oi = sum(opt.get('open_interest', 0)
                              for opt in call_data['results']
                              if opt['expiration_date'] == exp_date and opt['strike_price'] == strike)

                # Get implied volatility (using average of put and call IV)
                put_iv = next((opt.get('implied_volatility', 0)
                               for opt in put_data['results']
                               if opt['expiration_date'] == exp_date and opt['strike_price'] == strike), 0)

                call_iv = next((opt.get('implied_volatility', 0)
                                for opt in call_data['results']
                                if opt['expiration_date'] == exp_date and opt['strike_price'] == strike), 0)

                avg_iv = (put_iv + call_iv) / 2 if put_iv and call_iv else max(put_iv, call_iv)

                if avg_iv > 0:
                    # Calculate gamma using Black-Scholes
                    gamma = calculate_gamma(spot_price, strike, T, risk_free_rate, avg_iv, 'call')

                    # Calculate GEX for this strike
                    gex = (call_oi - put_oi) * gamma * spot_price * 100

                    gex_data.append({
                        'expiration_date': exp_date,
                        'strike_price': strike,
                        'put_oi': put_oi,
                        'call_oi': call_oi,
                        'implied_volatility': avg_iv,
                        'gamma': gamma,
                        'gex': gex
                    })

    # Convert to DataFrame and calculate totals
    df = pd.DataFrame(gex_data)
    if not df.empty:
        df['time_to_expiry'] = pd.to_datetime(df['expiration_date']) - pd.Timestamp(current_date)
        df['days_to_expiry'] = df['time_to_expiry'].dt.days

        # Calculate total GEX and aggregate by expiration
        total_gex = df['gex'].sum()
        gex_by_expiry = df.groupby('expiration_date')['gex'].sum()

        return {
            'detailed_data': df,
            'total_gex': total_gex,
            'gex_by_expiry': gex_by_expiry
        }

    return None


# Example usage
def main(ticker, api_key):
    try:
        results = calculate_gex(ticker, api_key)

        if results:
            print(f"\nResults for {ticker}:")
            print(f"Total GEX: {results['total_gex']:,.2f}")

            print("\nGEX by Expiration Date:")
            print(results['gex_by_expiry'])

            print("\nDetailed Options Data:")
            print(results['detailed_data'])

            # Optional: Save to CSV
            results['detailed_data'].to_csv(f'{ticker}_gex_analysis.csv', index=False)
            print(f"\nDetailed data saved to {ticker}_gex_analysis.csv")

    except Exception as e:
        print(f"Error calculating GEX: {str(e)}")


if __name__ == "__main__":
    API_KEY = POLYGON_API_KEY  # Replace with your API key
    TICKER = "SPY"  # Replace with desired ticker
    main(TICKER, API_KEY)