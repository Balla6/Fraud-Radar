import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N_CUST, N_MERCH, N_DEV, N_CARDS, N_TX = 2000, 300, 1500, 3000, 20000  # small & fast

def synthesize():
    base = pd.Timestamp('2024-01-01')
    customers = pd.DataFrame({
        'customer_id': np.arange(N_CUST),
        'signup_time': base + pd.to_timedelta(rng.integers(0,120,N_CUST), unit='D'),
        'country': rng.choice(['US','GB','CA','DE','FR','NG','IN'], N_CUST)
    })
    cards = pd.DataFrame({
        'card_id': np.arange(N_CARDS),
        'customer_id': rng.integers(0, N_CUST, N_CARDS),
        'brand': rng.choice(['VISA','MC','AMEX'], N_CARDS),
        'last4': [f"{rng.integers(0,9999):04d}" for _ in range(N_CARDS)],
        'active_flag': True
    })
    devices = pd.DataFrame({
        'device_id': np.arange(N_DEV),
        'os': rng.choice(['iOS','Android','Windows'], N_DEV),
        'user_agent_hash': rng.integers(10**6, 10**8, N_DEV).astype(str),
        'first_seen': base + pd.to_timedelta(rng.integers(0,150,N_DEV), unit='D')
    })
    merchants = pd.DataFrame({
        'merchant_id': np.arange(N_MERCH),
        'category': rng.choice(['grocery','travel','electronics','apparel','gaming','crypto'], N_MERCH),
        'country': rng.choice(['US','GB','CA','DE','FR','NG','IN'], N_MERCH),
        'risk_score': rng.uniform(0, 1, N_MERCH)
    })

    tx_time = base + pd.to_timedelta(rng.integers(0,150,N_TX), unit='D') \
                    + pd.to_timedelta(rng.integers(0,24*60,N_TX), unit='m')
    tx_time = pd.Series(tx_time).sort_values().values

    tx = pd.DataFrame({
        'tx_id': np.arange(N_TX),
        'card_id': rng.integers(0, N_CARDS, N_TX),
        'device_id': rng.integers(0, N_DEV, N_TX),
        'merchant_id': rng.integers(0, N_MERCH, N_TX),
        'amount': np.round(np.exp(rng.normal(3.0,1.0,N_TX)), 2),
        'currency': 'USD',
        'tx_time': tx_time,
        'geo_lat': rng.normal(40.0, 10.0, N_TX),
        'geo_lon': rng.normal(-74.0, 10.0, N_TX),
        'ip_hash': rng.integers(10**6, 10**8, N_TX).astype(str)
    })

    prob = 0.002 + 0.02*(merchants.set_index('merchant_id').loc[tx['merchant_id'], 'risk_score'].values)
    prob += 0.01*(tx['amount'].values > 200).astype(float)
    hours = pd.to_datetime(tx['tx_time']).dt.hour
    prob += 0.005*((hours >= 0) & (hours <= 4)).astype(float)
    dev_counts = tx.groupby('device_id').cumcount()
    prob += np.clip(dev_counts/40.0, 0, 0.05)
    is_fraud = rng.binomial(1, np.clip(prob, 0, 0.3))

    labels = pd.DataFrame({
        'tx_id': tx['tx_id'],
        'is_fraud': is_fraud,
        'label_time': pd.to_datetime(tx['tx_time']) + pd.to_timedelta(2, unit='D')
    })

    Path('data/raw').mkdir(parents=True, exist_ok=True)
    customers.to_csv('data/raw/customers.csv', index=False)
    cards.to_csv('data/raw/cards.csv', index=False)
    devices.to_csv('data/raw/devices.csv', index=False)
    merchants.to_csv('data/raw/merchants.csv', index=False)
    tx.to_csv('data/raw/transactions.csv', index=False)
    labels.to_csv('data/raw/labels.csv', index=False)
    print('Wrote 6 files to data/raw')

if __name__ == '__main__':
    synthesize()
