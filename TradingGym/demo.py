"""Demo: losowy agent na jednym dniu danych z animowanym wykresem.

Uruchomienie (z folderu TradingGym):  python demo.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trading_env

df = pd.read_csv('trading_env/test/data/SGXTWsample.csv', index_col=0, parse_dates=['datetime'])
env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price',
                       feature_names=['Price', 'Volume'])
env.reset()
done = False
while not done:
    state, reward, done, info = env.step(np.random.randint(3))  # losowy agent
    env.render()

print(info[info.position_variation != 0][['datetime', 'Price', 'position', 'reward']])
print('Wynik dnia:', round(info.reward.sum(), 2))
plt.ioff()
plt.show()
