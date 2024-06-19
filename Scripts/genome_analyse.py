import numpy as np
import pyreadr
import pandas as pd
import pyranges as pr
from hmmlearn import hmm

result = pyreadr.read_r('exampleForLukas.RDS')
df = result[None]
print(df.head())
print()

grouped = df.groupby(np.arange(len(df.index)) // 100, axis=0).max()


grouped['Signal'] = grouped['TT_S0'].apply(lambda x: 1 if x > 10 else 0)
signals = grouped['Signal'].to_numpy().reshape(-1, 1)
print(grouped.head())
print(signals)

hstates = np.random.randint(2, size = 70376)
print(hstates)


X_train = signals[:signals.shape[0] // 2]
X_validate = signals[signals.shape[0] // 2:]


np.random.seed(13)

model = hmm.CategoricalHMM(n_components=2, random_state=99, init_params='se')
model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
# print(model.transmat_.shape)
model.fit(signals)

predicted_states = model.predict(signals)

print(f'Transmission matrix Recovered:\n{model.transmat_.round(3)}\n\n')
print(f'Emission matrix Recovered:\n{model.emissionprob_.round(3)}\n\n')