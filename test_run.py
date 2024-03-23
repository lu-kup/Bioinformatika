import numpy as np
import pyreadr
import pandas as pd
import pyranges as pr
from hmmlearn import hmm


def get_hidden_states(offset, astuoniol):
    START_18 = 3000000
    END_18 = 91000000

    bins1 = list(np.arange(START_18 + offset, END_18, 100)) + [END_18]
    if offset != 0:
        bins1 = [START_18] + bins1

    print(bins1[:20], bins1[-20:])
    astuoniol['bin1'] = pd.cut(astuoniol['start'], bins=bins1)

    print(astuoniol.head())
    print(astuoniol.tail())

    sumos = astuoniol.groupby('bin1').sum()
    sumos.drop(['start', 'end'], axis=1, inplace=True)

    print(sumos[:20])


    signals = sumos['TT_S0'].to_numpy().reshape(-1, 1)
    print(signals)

    model = hmm.GaussianHMM(n_components=2, random_state=99, init_params='se')
    model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    model.fit(signals)

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    predicted_states = model.predict(signals)

    print(f'Transmission matrix Recovered:\n{model.transmat_.round(3)}\n\n')
    print(predicted_states)

    return predicted_states

def main():
    result = pyreadr.read_r('exampleForLukas.RDS')
    df = result[None]
    print(df.head())
    print()

    astuoniol = df[df['seqnames'] == '18']

    astuoniol.sort_values('start', inplace = True)

    res = []

    for i in range(20):
        current_offset = i * 5
        res.append(get_hidden_states(current_offset, astuoniol))

    result = np.asarray(res)

    print("RESULTS")
    print(result)

if __name__ == "__main__":
    main()