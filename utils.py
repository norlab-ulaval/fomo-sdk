from datetime import datetime

changeovertime = 1627387200 * 1e9
def get_num_times(bag, topics):
    times = [t for topic, msg, t in bag.read_messages(topics)]
    return len(times)

def get_start_week(rostime, gpstime):
    start_epoch = rostime * 1e-9
    dt = datetime.fromtimestamp(start_epoch)
    weekday = dt.isoweekday()
    if weekday == 7:
        weekday = 0  # Sunday
    g2 = weekday * 24 * 3600 + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    start_week = round(start_epoch - g2)
    hour_offset = round((gpstime - g2) / 3600)
    time_zone_offset = hour_offset * 3600.0        # Toronto time is GMT-4 or GMT-5 depending on time of year
    print('START WEEK: {} TIME ZONE OFFSET: {}'.format(start_week, time_zone_offset))
    return start_week, time_zone_offset
