import pyreadr
import pandas as pd
import pyranges as pr

result = pyreadr.read_r('exampleForLukas.RDS')
df = result[None]
print(df.head())
print()



df_19 = df[df['seqnames'] == '19']
print(df_19.head())
print()

df_3 = df[df['TT_S0'] > 3]
print(df_3.head())
print()

milijonas = df.iloc[5000000:6000000, 1]
#s0 = milijonas['TT_S0']
print(milijonas.head())

print(df.head())

df = df.rename(columns={'seqnames': 'Chromosome', 'start': 'Start', 'end': 'End', 'TT_S0' : 'Score'})
print("PYRANGES")
p = pr.PyRanges(df)
print(p)

hg19 = pr.data.chromsizes()
print(hg19)
pr.to_bigwig(p, "outpath.bw", hg19)