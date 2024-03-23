import numpy as np
import pyreadr
import pandas as pd
import pyranges as pr
from hmmlearn import hmm

result = pyreadr.read_r('exampleForLukas.RDS')
df = result[None]
print(df.head())
print()

astuoniol = df[df['seqnames'] == '18']

astuoniol.sort_values('start', inplace = True)

START_18 = 3000000
END_18 = 91000000

bins1 = list(np.arange(START_18, END_18, 100)) + [END_18]
bins2 = [START_18] + list(np.arange(START_18 + 5, END_18, 100)) + [END_18]
bins3 = [START_18] + list(np.arange(START_18 + 10, END_18, 100)) + [END_18]

print(bins1[:20], bins1[-20:])
print(bins2[:20], bins2[-20:])
print(bins3[:20], bins3[-20:])

astuoniol['bin1'] = pd.cut(astuoniol['start'], bins=bins1)
#astuoniol['bin2'] = pd.cut(astuoniol['start'], bins=bins2)
#astuoniol['bin3'] = pd.cut(astuoniol['start'], bins=bins3)

print(astuoniol.head())
print(astuoniol.tail())

sumos = astuoniol.groupby('bin1').sum()
sumos.drop(['start', 'end'], axis=1, inplace=True)

print(sumos[:20])