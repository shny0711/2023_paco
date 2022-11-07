import datetime

def nowstr():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y.%m.%d %H:%M:%S')