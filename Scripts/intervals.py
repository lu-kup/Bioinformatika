import pandas as pd
import numpy as np
import random
import string

# Generate random integers
integers = [random.randint(1, 100) for _ in range(10)]
integers2 = [_ + 1 for _ in integers]

# Generate random characters
characters = [''.join(random.choices(string.ascii_letters, k=1)) for _ in range(10)]

# Create DataFrame
df = pd.DataFrame({'Integers': integers, 'Integers2': integers2, 'Characters': characters})

print(df)

filtered = df.filter(regex="Int")
df['SUM'] = filtered.sum(axis=1)

df2 = df[df.columns.drop(list(df.filter(regex='Int')))]

df['Test'] = df['Integers'] > 50

y = df['Test']
print(y)

df["State_change"] = (df['Test'] != df['Test'].shift())
df_pokyciai = df[df["State_change"]]
df
print(df_pokyciai)

