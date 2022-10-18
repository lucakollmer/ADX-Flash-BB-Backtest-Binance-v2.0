# %% Preamble
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:31:50 2022

@author: lucak
"""

__version__ = '2.0'
__author__ = 'Luca Kollmer'

# %% Import Packages.

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# pip install python-binance.
from binance.client import Client
client = Client()

# Set pyplotlib styles.
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

# %% Request Price Series from Binance API and Transform to OHLCV.

def get_minute_data(symbol, timeframe, lookback):
    '''
    Request 'symbol' price series from Binance at 'timeframe' minute intervals
    over 'lookback' days. Transform price series to OHLCV with datetime 'Time'.

    Parameters
    ----------
    symbol : str
        Binance symbol of price feed e.g. 'BTCUSDT'.
    timeframe : int
        Interval between rows in price series in minutes.
    lookback : int
        Period in days to request data for.

    Returns
    -------
    df : pd.DataFrame
        Transformed price series with Time, OHLC, Volume columns.

    '''
    
    df = pd.DataFrame(
        client.get_historical_klines(symbol, 
                                     str(timeframe) + 'm',
                                     str(lookback) + ' days ago UTC'
                                     )
        )
    
    df = df.iloc[:,:6]
    
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open',
                                                         'High',
                                                         'Low',
                                                         'Close',
                                                         'Volume'
                                                         ]].astype(float)
    
    df.Time = pd.to_datetime(df.Time, unit='ms')
    
    return df

# %% Compute True Range Indicator on HLC Series.

def get_true_range(high, low, close):
    '''
    Calculates True Range for HLC price series.

    Parameters
    ----------
    high : float
    low : float
    close : float

    Returns
    -------
    df : pd.DataFrame
        One column dataframe containing True Range series.

    '''
    
    high_to_low = pd.DataFrame(high - low)
    
    close_to_high = pd.DataFrame(abs(high - close.shift(1)))
    
    close_to_low = pd.DataFrame(abs(low - close.shift(1)))
    
    true_range = [high_to_low, 
          close_to_high,
          close_to_low,
          ]
    
    df = pd.concat(true_range,
                   axis=1,
                   join='inner',
                   ).max(axis=1)
    
    return df

# %% Compute ATR and DMI Indicators on OHLC Series.

def get_ta_dmi(df, adx_period):
    '''
    Computes Average True Range (ATR) and Directional Movement Index (DMI)
    technical indicators on OHCL series. Returns OHLC series with additional
    'ATR', 'ADX', 'DIplus', 'DIminus' columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC series.
    adx_period : int
        Smoothing period for ADX.

    Returns
    -------
    df : pd.DataFrame
        Original df + four-columns: 'ATR', 'ADX', 'DIplus', 'DIminus'.

    '''
    
    df = df.copy()
    
    # Compute Average True Range.
    df['ATR'] = pd.DataFrame(
        get_true_range(df.High,
                       df.Low,
                       df.Close
                       )
        ).ewm(alpha=(1/adx_period),
                     min_periods=adx_period,
                     adjust=False
                     ).mean()
              
    # Compute Positive Directional Movement.
    df['+DM'] = np.where(df.High.diff() > -df.Low.diff(), 
                         df.High.diff(), 
                         0)
    
    df['+DM'] = np.where(df['+DM'] > 0, 
                         df['+DM'],
                         0)    
    
    # Compute Negative Directional Movement.
    df['-DM'] = np.where(-df.Low.diff() > df.High.diff(),
                         -df.Low.diff(),
                         0)
    
    df['-DM'] = np.where(df['-DM'] > 0
                         , df['-DM'],
                         0)
    
    # Compute Positive and Negative Directional Indices.
    df['DIplus'] = (100*(df['+DM'].ewm(alpha=(1/adx_period),
                                       min_periods=adx_period, 
                                       adjust=False
                                       ).mean())) / df['ATR']
    
    df['DIminus'] = (100*(df['-DM'].ewm(alpha=(1/adx_period),
                                        min_periods=adx_period,
                                        adjust=False
                                        ).mean())) / df['ATR']
    
    # Compute Average Directional Index.
    df['DX'] = df['DIplus'] + df['DIminus']
    
    df['DX'] = np.where(df['DX'] == 0, 
                        1,
                        df['DX'])
    
    df['Raw ADX'] = 100*abs(df['DIplus'] - df['DIminus']) / df['DX']
    
    df['ADX'] = df['Raw ADX'].ewm(alpha=(1/adx_period), 
                                  min_periods=adx_period, 
                                  adjust=False
                                  ).mean()

    # Drop Redundant Columns.
    df = df.drop(columns=['+DM', '-DM', 'Raw ADX', 'DX'])
    
    return df

# %% Compute Bollinger Bands (BB) Indicator on OHLC Series.

def get_ta_bb(df, bb_period):
    '''
    Computes Bollinger Bands (BB) technical indicators on OHCL series. Returns
    OHLC series with additional 'BBplus' and 'BBminus' columns.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    bb_period : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    df = df.copy()
    
    # Compute Bollinger Bands
    df['BBplus'] = df.Close.rolling(bb_period).mean() \
                   + 2*df.Close.rolling(bb_period).std(ddof=0)
                  
    df['BBminus'] = df.Close.rolling(bb_period).mean() \
                    - 2*df.Close.rolling(bb_period).std(ddof=0)

    return df 

# %% Function to Handle Division by Zero.

def division(numerator, denominator):
    '''
    Returns 0 if division of 'numerator' by 'denominator' does not exist.

    Parameters
    ----------
    numerator : number

    denominator : number


    Returns
    -------
    number
        Returns 'numberator'/'denominator' if it exists, else 0.

    '''
    
    return numerator / denominator if denominator else 0

# %% Engine to Perform Backtest of ADX Flash BB Strategy on Price Series.

def adx_flash_bb_strategy(df,
                          adx_threshold,
                          grace_period,
                          fee,
                          max_entries,
                          position_size,
                          cash_buffer,
                          adx_fill_tolerance,
                          entry_delay,
                          stop_multiplier
                          ):
    '''
    Rudimentary engine to iterate through price series and compute whether to
    open positions based on price closing outside of Bollinger Bands (BB), when
    an ADX Flash is incomplete. See www.lucakollmer.com for more information.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC series with additional ADX and BB data.
    adx_threshold : int
        ADX crossing this threshold generates ADX Flashes.
    grace_period : int
        Period after ADX Flash wherein opening positions is prohibited.
    fee : float
        Broker commission per trade.
    max_entries : int
        Maximum number of positions allowed per ADX Flash.
    position_size : float
        Proportion of account value assigned to each position.
    cash_buffer : float
        Proportion of account value prohibited from opening positions.
    adx_fill_tolerance : float
        Proportion of distance between position entry and ADX Flash open to
        set take profit.
    entry_delay : int
        Period after a position is opened wherein opening positions is 
        prohibited.
    stop_multiplier : int
        Multiple of distance from position entry and ADX Flash open to set 
        position stop loss.

    Returns
    -------
    df : pd.DataFrame
        Price series with additional columns tracking strategy performance.
    complete_flashes_df : pd.DataFrame
        DESCRIPTION.
    active_flashes_df : pd.DataFrame
        DESCRIPTION.
    closed_positions_df : pd.DataFrame
        DESCRIPTION.
    open_positions_df : pd.DataFrame
        DESCRIPTION.

    '''
    
    df = df.copy()
    
    # Drop first 200 rows where ADX is inaccurate.
    df = df.iloc[200:, :]
    
    # Generate ADX Flash signals when ADX crosses 'adx_threshold'.
    # ADX Flash starts when 'Flash' = 1 and ends 'Flash' = -1.
    df['ADXsignal'] = np.where(df['ADX'] < adx_threshold, 
                               1,
                               0)
    
    df['Flash'] = np.where(df['ADXsignal'].diff() == -1, 
                           -1, 
                           0)
    
    df['Flash'] = np.where(df['ADXsignal'].diff() == 1, 
                           1, 
                           df['Flash'])
    
    # Generate series columns to track strategy performance.
    df['Account'] = 0
    df['Cash'] = 0
    df['Utilisation'] = 0.0
    df['Longs'] = 0
    df['Shorts'] = 0
    df['Positions'] = 0
    
    # Initiate local variables.
    positions = 0
    longs = 0
    shorts = 0
    max_positions = 0
    account = 100000
    account_start = account
    cash = account
    account_ath = account
    max_drawdown = 0
    trades = 0
    total_longs = 0
    total_shorts = 0
    
    # Initiate empty lists to track ADX Flashes and positions.
    active_flashes = []
    complete_flashes = []
    current_flash = []
    open_positions = []
    closed_positions = []
    
    # Iterate row-wise through price series.
    for row in tqdm(df.itertuples()):
        
        # Update performance series.
        max_positions = max(positions, max_positions)
        account_ath = max(account, account_ath)
        drawdown = (account_ath - account) / account_ath
        max_drawdown = max(max_drawdown, drawdown)

        # Update Account and Cash balance.
        df.at[row.Index, 'Account'] = account
        df.at[row.Index, 'Cash'] = cash
        df.at[row.Index, 'Longs'] = longs
        df.at[row.Index, 'Shorts'] = shorts
        df.at[row.Index, 'Positions'] = positions
        df.at[row.Index, 'Utilisation'] = (account - cash) / account
        
        # Check Active ADX Flashes #
        
        # Iterate through active_flashes to check status of active Flashes.
        for flash in active_flashes:
            
            flash[4] += 1
            flash[11] += 1
            
            # Update maximum return of most recent active Flash.
            if (active_flashes.index(flash) == len(active_flashes) - 1):
                
                if (row.High < flash[2]):
                    
                    flash[5] = min([(row.Low - flash[2]) / flash[2],
                                    flash[5]])
                    
                elif (row.Low > flash[2]):
                    flash[5] = max([(row.High - flash[2]) / flash[2], 
                                    flash[5]])
            
            # Check if price retraces to open of bar that Flash ended.
            if (flash[4] > grace_period 
                and not (row.High < flash[2] 
                         or row.Low > flash[2])
                ):
                
                complete_flashes.append(
                    active_flashes.pop(
                        active_flashes.index(flash)))
                
        # Check Open Positions #
        
        # Iterate through open positions.
        for position in open_positions:
            
            position[4] += 1
            
            # Check if price has reached position take profit.
            if not (row.High < position[3] or row.Low > position[3]):
                
                position[5] = 1
                positions -= 1
    
                if (position[1] == 1):
                    longs -= 1
                    
                elif (position[1] == -1):
                    shorts -= 1
    
                # Credit cash balance with position return.
                account += position[9]
                cash += position[7] + position[9]
                
                closed_positions.append(
                    open_positions.pop(
                        open_positions.index(position)))
                
            # Check if price has reached position stop loss.
            elif (position[2] > position[10] 
                  and row.Low < position[10]
                  or position[2] < position[10] 
                  and row.High > position[10]):
                
                position[5] = -1
                positions -= 1
                
                if (position[1] == 1):
                    longs -= 1
                    
                elif (position[1] == -1):
                    shorts -= 1
            
                # Credit cash balance with position return.
                account += position[12]
                cash += position[7] + position[12]

                closed_positions.append(
                    open_positions.pop(
                        open_positions.index(position)))
                
        # Create New ADX Flash #
        
        # New ADX Flash opens (ADX crosses below adx_threshold).
        if (row.Flash == 1):
            
            current_flash = [row.Time, 0]
            
            if (len(active_flashes) > 0):
                
                active_flashes[len(active_flashes) - 1][7] = 0
                
        # New ADX Flash closes (ADX crosses above adx_threshold).
        if (len(current_flash) > 0 
            and row.ADXsignal == 1):
            
            current_flash[1] += 1
            
        if (len(current_flash) > 0 
            and row.Flash == -1):
            
            current_flash.append(row.Open)
            
            # Use DMI to set initial direction of ADX Flash.
            if (row.DIplus > row.DIminus):
                
                current_flash.append(1)
                
            else:
                
                current_flash.append(-1)
            
            # Initiate ADX Flash variables:
            #  4 = age,
            #  5 = high/low,
            #  6 = entries,
            #  7 = entries allowed,
            #  8 = size,
            #  9 = entry,
            # 10 = take profit,
            # 11 = bars since last entry.
            for i in range(8):
                
                current_flash.append(0)
                
            current_flash[7] = 1
            active_flashes.append(current_flash)
            current_flash = []        
            
        # Create New Position #
        
        # Open Position.
        if (len(active_flashes) > 0):
            
            x = len(active_flashes) - 1
            
            if (active_flashes[x][4] > grace_period 
                and active_flashes[x][7] == 1):
                
                flashopen = active_flashes[x][2]
                bias = 0
                
                # Check move is greater than fee to open and close position
                # and check Flash entries < max_entries
                # and cash > buffer*account
                # and bars since last entry > entry delay
                if (abs((row.Close - flashopen) / flashopen) > fee * 2 
                    and active_flashes[x][6] < max_entries 
                    and cash - (account*position_size) > cash_buffer*account 
                    and active_flashes[x][11] >= entry_delay):
                     
                    # Determine bias of entry
                    if (row.Close < flashopen 
                        and row.Close < row.BBminus):
                        
                        bias = 1
                        longs += 1
                        total_longs += 1
                        
                    elif (row.Close > flashopen 
                          and row.Close > row.BBplus):
                        
                        bias = -1
                        shorts += 1
                        total_shorts += 1
                    
                    if (bias != 0):
                        
                        # Calculate position variables. 
                        # Assumes entry at bar close.
                        cost_basis = account*position_size
                        position_size = bias*(cost_basis/(1+fee))/row.Close
                        tp = (flashopen - row.Close)*adx_fill_tolerance \
                             + row.Close
                        proceeds = (position_size*tp)*(1-fee)
                        tp_return = proceeds + bias*(-1)*cost_basis
                        sl = row.Close \
                            - stop_multiplier*(flashopen - row.Close)#*tolerance
                        sl_proceeds = (position_size*sl)*(1-fee)
                        loss = sl_proceeds + bias*(-1)*cost_basis

                        # Initiate position variables:
                        #  0 = Time,
                        #  1 = Long/Short,
                        #  2 = Entry,
                        #  3 = Take Profit,
                        #  4 = Age,
                        #  5 = Status,
                        #  6 = Position Size,
                        #  7 = Cost Basis,
                        #  8 = Proceeds (if sold Short),
                        #  9 = Take Profit Return,
                        # 10 = Stop Loss,
                        # 11 = Stop Loss Proceeds,
                        # 12 = Stop Loss Return.
                        open_positions.append([row.Time, 
                                               bias, 
                                               row.Close, 
                                               tp, 
                                               0, 
                                               0,
                                               position_size, 
                                               cost_basis, 
                                               proceeds,
                                               tp_return, 
                                               sl, 
                                               sl_proceeds, 
                                               loss])
                        
                        active_flashes[x][9] = (active_flashes[x][9]  \
                                                * active_flashes[x][6] 
                                                + row.Close) \
                                                / (active_flashes[x][6] + 1)
                                                
                        active_flashes[x][10] = (active_flashes[x][10] \
                                                 * active_flashes[x][6] \
                                                 + tp) \
                                                 / (active_flashes[x][6] + 1)
                                                 
                        active_flashes[x][6] += 1
                        active_flashes[x][8] += position_size
                        active_flashes[x][9] = active_flashes[x][9]\
                                               / active_flashes[x][6]
                        
                        # Update global variables.
                        active_flashes[x][11] = 0
                        positions += 1
                        trades += 1
                        cash -= cost_basis 
    
    # Output #
    
    # Plots #
    
    # Price series plot.
    ax1 = plt.subplot2grid((23,1), 
                           (0,0), 
                           rowspan=5, 
                           colspan=1
                           )
    
    ax1.plot(df['Close'],
             linewidth=1, 
             color='#ff9800', 
             alpha=1
             )
    
    ax1.set_title('BTCUSDT Closing Price')

    # Position plot.    
    ax2 = plt.subplot2grid((23,1), 
                           (6,0), 
                           rowspan=5, 
                           colspan=1
                           )
    
    ax2.plot(df['Positions'], 
             label='Positions', 
             linewidth=1, 
             color='#000000', 
             alpha=0.5
             )
    
    ax2.plot(df['Longs'], 
             label='Longs', 
             linewidth=1, 
             color='#00FF00', 
             alpha=0.8
             )
    
    ax2.plot(df['Shorts'], 
             label='Shorts', 
             linewidth=1, 
             color='#FF0000',
             alpha=0.8
             )    
    
    ax2.legend()
    
    # Equity curve plot.
    ax3 = plt.subplot2grid((23,1), 
                           (12,0), 
                           rowspan=5, 
                           colspan=1
                           )
    
    ax3.plot(df['Account'], 
             color='#26a69a', 
             label='Account',
             linewidth=3, 
             alpha=0.8
             )

    ax3.legend() 
    
    # Cash utilisation plot.
    ax4 = plt.subplot2grid((23,1), 
                           (18,0), 
                           rowspan=5, 
                           colspan=1
                           )
    
    ax4.plot(df['Utilisation'],
             color='#ff0000', 
             label='Utilisation', 
             linewidth=1, 
             alpha=0.8
             )
    
    ax4.legend()
    
    # Convert lists to dataframes.
    complete_flashes_df = pd.DataFrame(complete_flashes, 
                                       columns=['Time', 
                                                'Len', 
                                                'Flash', 
                                                'DI', 
                                                'Age', 
                                                'Move', 
                                                'Entries',
                                                'Entries Allowed', 
                                                'Size', 
                                                'Entry', 
                                                'TP', 
                                                'Last Entry']
                                       )
    
    complete_flashes_df = complete_flashes_df.drop(columns=['Entries Allowed',
                                                            'Last Entry'])
    
    active_flashes_df = pd.DataFrame(active_flashes, 
                                     columns=['Time', 
                                              'Len', 
                                              'Flash', 
                                              'DI', 
                                              'Age', 
                                              'Move', 
                                              'Entries',
                                              'Entries Allowed', 
                                              'Size', 
                                              'Entry', 
                                              'TP', 
                                              'Last Entry']
                                     )
    
    active_flashes_df = active_flashes_df.drop(columns=['Entries Allowed', 
                                                        'Last Entry'])
    
    #Positions
    open_positions_df = pd.DataFrame(open_positions,
                                     columns=['Time', 
                                              'Bias', 
                                              'Entry', 
                                              'TP', 
                                              'Age', 
                                              'Status', 
                                              'Size',
                                              'CB', 
                                              'Proceeds', 
                                              'Return', 
                                              'SL', 
                                              'SL Proceeds', 
                                              'Loss']
                                     )
    
    open_positions_df = open_positions_df.drop(columns=['Status'])
    
    closed_positions_df = pd.DataFrame(closed_positions,
                                      columns=['Time', 
                                               'Bias', 
                                               'Entry', 
                                               'TP', 
                                               'Age', 
                                               'Status', 
                                               'Size',
                                               'CB', 
                                               'Proceeds',
                                               'Return', 
                                               'SL', 
                                               'SL Proceeds', 
                                               'Loss']
                                      )    
    
    # Drop redundant columns from price series.
    df = df.drop(columns=['ADX', 'ATR', 'High', 'Low', 'Volume', 'BBplus', 
                          'BBminus', 'DIplus', 'DIminus', 'ADXsignal']
                 )
    
    # Collect performance statistics.
    benchmark = (df['Close'].iloc[-1] - df['Close'].iloc[0]) \
                /df['Close'].iloc[0]
                
    total_return = (account - account_start)/account_start
    
    wins = losses = long_wins = long_losses = short_wins = short_losses = 0
    
    # Count long and short wins.
    for i in closed_positions_df.itertuples():
        
        if (i.Bias == 1 and i.Status == 1):
            
            long_wins += 1
            wins += 1
            
        elif (i.Bias == 1 and i.Status == -1):
            
            long_losses += 1
            losses += 1
            
        elif (i.Bias == -1 and i.Status == 1):
            
            short_wins += 1
            wins += 1
            
        elif (i.Bias == -1 and i.Status == -1):
            
            short_losses += 1
            losses += 1  
            
    # Print Performance Statistics #
    
    print(max_positions)
    
    # Print backtest parameters.
    print('Flash Threshold: ' + str(adx_threshold))
    print('Grace Period: ' + str(grace_period))
    print('Trading Fee: ' + str(fee))
    print('Max Entries per Flash: ' + str(max_entries))
    print('Position Size: ' + str(position_size))
    print('Cash Buffer: ' + str(cash_buffer))
    print('TP Tolerance: ' + str(adx_fill_tolerance))
    print('Delay between Entries: ' + str(entry_delay))
    print('Risk/Reward: ' + str(stop_multiplier) + ':1')
    print()
    
    # Print performance statistics.
    print('Benchmark: ' + str(round(benchmark, 4)))
    print('Return: ' + str(round(total_return, 4)))
    print('Alpha: ' + str(round(total_return - benchmark, 4)))
    print('Max Return: ' + str(round((account_ath-account_start)/account_start,
                                     4)))
    
    print('Max Drawdown: ' + str(round(max_drawdown, 4)))
    print('Utilisation: ' + str(round(df['Utilisation'].max(), 4)))
    print()
    
    # Print summary position data.
    print('Trades: ' + str(trades) + ', Wins: ' + str(wins) + ', Losses: ' \
          + str(losses) + ', Win Rate: ' + str(round(division(wins, losses),
                                                     2)))
        
    print('Longs: ' + str(total_longs) + ', Wins: ' + str(long_wins) \
          + ', Losses: ' + str(long_losses) + ', Win Rate: ' \
          + str(round(division(long_wins, long_losses), 2)))
        
    print('Shorts: ' + str(total_shorts) + ', Wins: ' + str(short_wins) \
          + ', Losses: ' + str(short_losses) + ', Win Rate: ' \
          + str(round(division(short_wins, short_losses), 2)))    
    
    # Returns all dataframes for troubleshooting and further analysis.
    return (df, 
            complete_flashes_df, 
            active_flashes_df, 
            closed_positions_df, 
            open_positions_df
            )
                        
# %% Test
test = get_ta_dmi(get_ta_bb(get_minute_data('BTCUSDT', 1, 30), 20), 14)
test2 = adx_flash_bb_strategy(test, 12, 7, 0.001, 3, 0.05, 0.05, 0.95, 3, 100)

