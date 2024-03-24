START_18 = 3000000
END_18 = 91000000

def add_bins(offset, astuoniol):
    breaks = list(np.arange(START_18 + offset, END_18, 100))
    from_array = [START_18] + breaks
    to_array = breaks + [END_18]

    bins = pd.IntervalIndex.from_arrays(from_array, to_array)

    print(bins)

    astuoniol['bin_offset_' + str(offset)] = pd.cut(astuoniol['start'], bins=bins)

    print(astuoniol.head())
    print(astuoniol.tail())

    return bins