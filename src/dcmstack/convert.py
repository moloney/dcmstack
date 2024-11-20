"""Provide some common conversions for DICOM elements"""
import re
from datetime import date, datetime, timedelta, timezone


def tm_to_sec(time_str):
    '''Convert a DICOM time value (value representation of 'TM') to the number
    of seconds past midnight.

    Parameters
    ----------
    time_str : str
        The DICOM time value string

    Returns
    -------
    A floating point representing the number of seconds past midnight
    '''
    #Allow ACR/NEMA style format by removing any colon chars
    time_str = time_str.replace(':', '')
    #Only the hours portion is required
    result = int(time_str[:2]) * 3600
    str_len = len(time_str)
    if str_len > 2:
        result += int(time_str[2:4]) * 60
    if str_len > 4:
        result += float(time_str[4:])
    return float(result)


def da_to_date(date_str):
    '''Convert a DICOM date value (value representation of 'DA') to a python `date`

    Parameters
    ----------
    date_str : str
        The DICOM date value string
    '''
    #Allow ACR/NEMA style format by removing any dot chars
    date_str = date_str.replace('.', '')
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))


def dt_to_datetime(dt_str):
    '''Convert a DICOM datetime value (value representation of 'DT') to a python `datetime`

    Parameters
    ----------
    dt_str : str
        The DICOM datetime value string
    '''
    dt_str, suffix = re.match("([0-9\.]+)([\+\-][0-9]{4})?", dt_str).groups()
    year = int(dt_str[:4])
    month = day = 1
    hour = minute = seconds = microsecs = 0
    if len(dt_str) > 4:
        month = int(dt_str[4:6])
    if len(dt_str) > 6:
        day = int(dt_str[6:8])
    if len(dt_str) > 8:
        hour = int(dt_str[8:10])
    if len(dt_str) > 10:
        minute = int(dt_str[10:12])
    if len(dt_str) > 12:
        seconds = int(dt_str[12:14])
    if len(dt_str) > 14:
        microsecs = int(float(dt_str[14:]) * 1e6)
    tzinfo = None
    if suffix is not None:
        td = timedelta(hours=int(suffix[1:3]), minutes=int(suffix[3:5]))
        if suffix[0] == '-':
            td *= -1
        tzinfo = timezone(td)
    return datetime(year, month, day, hour, minute, seconds, microsecs, tzinfo=tzinfo)
