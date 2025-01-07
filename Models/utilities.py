from datetime import datetime, timedelta




def time_to_minutes(time_str):
    """Convert HH:MM time format to total minutes."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(minutes):
    """Convert total minutes to HH:MM time format."""
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}:{minutes:02}"

def average_time(time_list):
    """Calculate the average time from a list of HH:MM time strings."""
    total_minutes = sum(time_to_minutes(time) for time in time_list)
    average_minutes = total_minutes / len(time_list)
    return minutes_to_time(int(average_minutes))
def max_time(time_list):
    """Calculate the maximum time from a list of HH:MM time strings."""
    max_minutes = max(time_to_minutes(time) for time in time_list)
    return minutes_to_time(int(max_minutes))

def format_time( sim_time):
    hours = int(sim_time)
    minutes = int((sim_time - int(sim_time)) * 60)
    return f"{hours}:{minutes:02}"

def time_date(year , month , day , hour , minute =0):
    current_time = datetime(year, month, day, hour)
    current_time = f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d} {hour}:{minute:02d}"
    return current_time
def get_closest_time(current_time):
    """
    Get the closest time to the current time where the hour is a multiple of 3 and the minutes are zero.
    
    Parameters:
    current_time (datetime): The current time.
    
    Returns:
    datetime: The closest time with hour % 3 == 0 and minutes == 0.
    """
    current_hour = current_time.hour
    next_hour = (current_hour // 3 + 1) * 3 if current_hour % 3 != 0 else current_hour
    if next_hour == 24:
        next_hour = 0
        current_time += timedelta(days=1)
    closest_time = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    return closest_time
def get_closest_time(current_time):
    """
    Get the closest time to the current time where the hour is a multiple of 3 and the minutes are zero.
    
    Parameters:
    current_time (datetime): The current time.
    
    Returns:
    datetime: The closest time with hour % 3 == 0 and minutes == 0.
    """
    current_hour = current_time.hour
    
    # Calculate the next closest time
    next_hour = (current_hour // 3 + 1) * 3 if current_hour % 3 != 0 else current_hour
    if next_hour == 24:
        next_hour = 0
        next_time = (current_time + timedelta(days=1)).replace(hour=next_hour, minute=0, second=0, microsecond=0)
    else:
        next_time = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    
    # Calculate the previous closest time
    prev_hour = (current_hour // 3) * 3 if current_hour % 3 != 0 else current_hour - 3
    if prev_hour < 0:
        prev_hour = 21
        prev_time = (current_time - timedelta(days=1)).replace(hour=prev_hour, minute=0, second=0, microsecond=0)
    else:
        prev_time = current_time.replace(hour=prev_hour, minute=0, second=0, microsecond=0)
    
    # Determine which of the two times is closer to the current time
    if abs((current_time - prev_time).total_seconds()) <= abs((next_time - current_time).total_seconds()):
        return prev_time
    else:
        return next_time