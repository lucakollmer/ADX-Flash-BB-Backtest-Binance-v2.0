# ADX-Flash-BB-Backtest-Binance-v2.0

Version 2.0

Comprehensive backtest of my ADX Flash BB Strategy on a single Binance symbol (BTCUSDT).
Models longs and shorts.

Output from one year of minute data:

Flash Threshold: 12
Grace Period: 7
Trading Fee: 0.001
Max Entries per Flash: 3
Position Size: 0.05
Cash Buffer: 0.05
TP Tolerance: 0.95
Delay between Entries: 3
Risk/Reward: 100:1

Benchmark: -0.2108
Return: 0.4806
Alpha: 0.6914
Max Return: 0.4806
Max Drawdown: 0.1012
Utilisation: 0.9497

Trades: 1913, Wins: 1889, Losses: 14, Win Rate: 134.93
Longs: 963, Wins: 952, Losses: 7, Win Rate: 136.0
Shorts: 950, Wins: 937, Losses: 7, Win Rate: 133.86

Strategy blows up over longer time periods owing to huge risk reward; stops are set too wide so losses are too great. 
Conclusion: cryptocurrencies are too volatile for this strategy so will move onto FX.
