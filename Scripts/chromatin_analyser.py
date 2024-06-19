import numpy as np
import pyreadr
import pandas as pd
import pyranges as pr
from hmmlearn import hmm

START_18 = 3000000
END_18 = 91000000
CHROMOSOME_NO = '18'

def add_bins(offset, chromosome):
    breaks = list(np.arange(START_18 + offset, END_18, 100))
    from_array = [START_18] + breaks
    to_array = breaks + [END_18]

    bins = pd.IntervalIndex.from_arrays(from_array, to_array)

    print("BINS", bins)

    chromosome['bin_offset_' + str(offset)] = pd.cut(chromosome['start'], bins=bins)

    print("HEAD")
    print(chromosome.head())
    print("TAIL")
    print(chromosome.tail())

    return bins

def get_hidden_states(offset, chromosome):

    sumos = chromosome.groupby('bin_offset_' + str(offset)).sum()
    sumos.drop(['start', 'end'], axis=1, inplace=True)

    print("SUMOS")
    print(sumos)


    signals = sumos['TT_S0'].to_numpy().reshape(-1, 1)
    print(signals)

    model = hmm.GaussianHMM(n_components=2, random_state=99, init_params='se')
    model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    model.fit(signals)

    predicted_states = model.predict(signals)

    print(f'Transmission matrix Recovered:\n{model.transmat_.round(3)}\n\n')
    print(predicted_states[:50])

    return predicted_states

def main():
    result = pyreadr.read_r('../Data/exampleForLukas.RDS')
    df = result[None]
    print(df.head())
    print(df["start"].min())
    print(df["start"].max())
    print()

    chromosome = df[df['seqnames'] == CHROMOSOME_NO]

    chromosome.sort_values('start', inplace = True)

    res = []

    for i in range(20):
        current_offset = i * 5
        bins = add_bins(current_offset, chromosome)
        predicted_states = get_hidden_states(current_offset, chromosome)
        states_df = pd.DataFrame(predicted_states, index=bins, columns = ['predicted_state_' + str(current_offset)])
        print(states_df[:50])
        print("JOIN")
        chromosome = chromosome.join(states_df, on = 'bin_offset_' + str(current_offset))
        print(chromosome[50:100])

    filtered = chromosome.filter(regex="predicted_state_")
    chromosome['predict_state_SUM'] = filtered.sum(axis=1)

    print(chromosome)

    chromosome = chromosome[chromosome.columns.drop(list(chromosome.filter(regex='bin_offset_')))]
    chromosome = chromosome[chromosome.columns.drop(list(chromosome.filter(regex='predicted_state_')))]

    chromosome.to_pickle('Data/out2.pkl')

    #result = np.asarray(res)

    #print("RESULTS")
    #print(result)

if __name__ == "__main__":
    main()