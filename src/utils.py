
def roundup(x):
    """Round to the nearest 50."""
    return x if x % 50 == 0 else x + 50 - x % 50