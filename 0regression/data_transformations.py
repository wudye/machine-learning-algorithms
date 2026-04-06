from statistics import mean, stdev

def normalize(data, ndigits=3):
    """Normalize the data"""
    x_min = min(data)
    x_max = max(data)
    return [round((x - x_min) / (x_max - x_min), ndigits) for x in data]

def standardize(data, ndigits=3):
    """Standardize the data"""
    x_mean = mean(data)
    x_std = stdev(data)
    return [round((x - x_mean) / x_std, ndigits) for x in data]
