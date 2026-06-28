import numpy as np
import datetime as dtime
import numba as nb
from math import pi as pi

class Constants:
    KER_YEAR_DAY = 426
    KER_DAY_HOUR = 6
    KER_HOUR_MIN = 60
    KER_MIN_SEC = 60
    KER_YEAR_SEC = 9201600.0
    KER_DAY_SEC = 21600.0

def get_ut_secs(year: int, day: int, hour: int, minute: int, seconds: int, ker_time = False):
    """Calculate elapsed seconds since the epoch tracker.

    Uses the KSP savefile start for Kerbal time, or the J2000 epoch 
    for real-world time.

    Args:
        year: Year of the target timestamp.
        day: Day of the year (1-indexed).
        hour: Hour of the day (0-23).
        minute: Minute of the hour (0-59).
        seconds: Seconds of the minute (0-59).
        ker_time: True to use Kerbin calendar rules, False for Earth.

    Returns:
        float: Total elapsed seconds.
    """
    
    if ker_time:
        elapsed_years = year - 1
        elapsed_days = day - 1
        
        time_conversions = (
            (elapsed_years, Constants.KER_YEAR_SEC),
            (elapsed_days, Constants.KER_DAY_SEC),
            (hour, Constants.KER_HOUR_MIN * Constants.KER_MIN_SEC),
            (minute, Constants.KER_MIN_SEC),
            (seconds, 1)
        )

        return float(sum(val * const for val, const in time_conversions))
    
    else:
        j2000_epoch = dtime.datetime(2000, 1, 1, 12, 0, 0, tzinfo = dtime.timezone.utc)
        base_date = dtime.datetime(year, 1, 1, tzinfo = dtime.timezone.utc)

        target_date = base_date + dtime.timedelta(day - 1, hours = hour, minutes = minute, seconds = seconds)

        return (target_date - j2000_epoch).total_seconds()