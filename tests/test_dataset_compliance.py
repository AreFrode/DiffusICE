def check_keys(store):
    assert 'samples' in store, "Missing 'samples' key"
    assert 'labels' in store, "Missing 'labels' key"

def check_normalization_csv(csv_path, channels):
    import pandas
    df = pandas.read_csv(csv_path, index_col = 0)
    assert 'mean_values' in df.columns, "CSV missing 'mean_values' column"
    assert 'std_values' in df.columns, "CSV missing 'std_values' column"

    missing = [c for c in channels if c not in df.index]
    assert not missing, f"Channels missing from normalized CSV: {missing}"