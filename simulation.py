import simpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta

import simpy
import numpy as pd
import pandas as pd
import random

def time_to_minutes(time_str):
    """Convert HH:MM time format to total minutes."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(minutes):
    """Convert total minutes to HH:MM time format."""
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}"

def average_time(time_list):
    """Calculate the average time from a list of HH:MM time strings."""
    total_minutes = sum(time_to_minutes(time) for time in time_list)
    average_minutes = total_minutes / len(time_list)
    return minutes_to_time(int(average_minutes))

import simpy
import numpy as np
import pandas as pd
import pandas as pd
import random

class ShipArrivalModel:
    def __init__(self, env, fleet_composition, production_rate_per_year, initial_eta):
        self.env = env
        self.fleet_composition = fleet_composition  # Ship types and probabilities
        self.production_rate_per_year = production_rate_per_year  # Theoretical production rate (m³/year)
        self.production_rate_per_day = production_rate_per_year / 365  # Convert to daily rate
        self.previous_eta = initial_eta  # Initial ETA for the first ship

    def select_random_ship(self):
        """Randomly select a ship type based on fleet composition probabilities."""
        ship_type = np.random.choice(self.fleet_composition['Ship_Type'], 
                                     p=self.fleet_composition['Composition'])
        capacity = self.fleet_composition.loc[self.fleet_composition['Ship_Type'] == ship_type, 'Capacity'].values[0]
        return ship_type, capacity

    def calculate_iat(self, capacity):
        """Calculate the Inter-Arrival Time (IAT) for the ship."""
        iat = capacity / self.production_rate_per_day
        return iat

    def calculate_eta(self, iat):
        """Calculate the ETA of the current ship by adding IAT to the previous ETA."""
        eta = self.previous_eta + iat
        return eta

    def calculate_ata(self, iat):
        """Calculate the Actual Time of Arrival (ATA) using a triangular distribution."""
        ata = self.previous_eta + np.random.triangular(0.5 * iat, iat, 1.5 * iat)
        return ata

    
class GenerateShips:
    def __init__(self, env, berth_type1, berth_type2, channel, weather_data, start_time, mean_pre_service_time, mean_post_service_time, stats, ship_arrival_model):
        self.env = env
        self.berth_type1 = berth_type1  
        self.berth_type2 = berth_type2
        self.weather_data = weather_data
        #self.mean_arrival_number = mean_arrival_number
        self.mean_pre_service_time = mean_pre_service_time
        self.mean_post_service_time = mean_post_service_time
        self.start_time = start_time
        self.stats = stats
        self.channel = channel
        self.ship_arrival_model = ship_arrival_model  # Reference to the ShipArrivalModel instance

    def generate_ships(self):
        while True:
            # Use the ship arrival model to select a ship and calculate arrival times
            ship_type, ship_capacity = self.ship_arrival_model.select_random_ship()
            iat = self.ship_arrival_model.calculate_iat(ship_capacity)
            eta = self.ship_arrival_model.calculate_eta(iat)
            ata = self.ship_arrival_model.calculate_ata(iat)

            # Update the model's previous ETA to the current one
            self.ship_arrival_model.previous_eta = eta
            # Generate the ship process in the environment
            yield self.env.timeout(ata)  # Ship arrives at ATA
            ship = Ship(self.env,ship_type , self.berth_type1, self.berth_type2, self.channel, self.weather_data, self.start_time, self.mean_pre_service_time, self.mean_post_service_time, self.stats)
            self.env.process(ship.process())

class Stats:
    def __init__(self):
        self.total_times = []
        self.berth_waiting_times = []
        self.mc_waiting_times = []
        self.processing_times = []
        self.waiting_times = []
        self.queue_sizes = {"berth_type1": [], "berth_type2": [], "berth": []}
        self.ship_data = []
        self.forced_departures = 0
        self.cargo_volume = {"exported": 0, "imported": 0}

    def add_total_time(self, time):
        self.total_times.append(time)

    def add_berth_waiting_time(self, time):
        self.berth_waiting_times.append(time)

    def add_mc_waiting_time(self, time):
        self.mc_waiting_times.append(time)

    def add_processing_time(self, time):
        self.processing_times.append(time)

    def add_waiting_time(self, time):
        self.waiting_times.append(time)

    def add_queue_size(self, queue_type, size):
        self.queue_sizes[queue_type].append(size)

    def add_ship_data(self, arrival, vessel, wt_meteorological_constraints, wt_ships_in_queue, wt_berth_occupied, processing_time, tat, expected_tat, departure_time):
        self.ship_data.append({
            "Arrival": arrival,
            "Vessel": vessel,
            "WT meteorological constraints": wt_meteorological_constraints,
            "WT Ships in queue": wt_ships_in_queue,
            "WT Berth occupied": wt_berth_occupied,
            "Processing Time": processing_time,
            "TAT": tat,
            "Expected TAT": expected_tat,
            "Departure_Time": departure_time
        })

    def increment_forced_departures(self):
        self.forced_departures += 1

    def add_cargo_volume(self, volume, direction):
        if direction == "exported":
            self.cargo_volume["exported"] += volume
        elif direction == "imported":
            self.cargo_volume["imported"] += volume

    def get_max_kpis(self):
        max_berth_waiting_time = max(self.berth_waiting_times, default=0)
        max_mc_waiting_time = max(self.mc_waiting_times, default=0)
        max_waiting_time = max(self.waiting_times, default=0)
        max_processing_time = max(self.processing_times, default=0)
        max_total_time = max(self.total_times, default=0)
        return {
            "max_berth_waiting_time": max_berth_waiting_time,
            "max_mc_waiting_time": max_mc_waiting_time,
            "max_waiting_time": max_waiting_time,
            "max_processing_time": max_processing_time,
            "max_total_time": max_total_time
        }

    def get_average_kpis(self):
        avg_berth_waiting_time = average_time(self.berth_waiting_times)
        avg_mc_waiting_time = average_time(self.mc_waiting_times)
        avg_waiting_time = average_time(self.waiting_times)
        avg_processing_time = average_time(self.processing_times)
        avg_total_time = average_time(self.total_times)
        return {
            "avg_berth_waiting_time": avg_berth_waiting_time,
            "avg_mc_waiting_time": avg_mc_waiting_time,
            "avg_waiting_time": avg_waiting_time,
            "avg_processing_time": avg_processing_time,
            "avg_total_time": avg_total_time
        }

    def get_ship_type_kpis(self):
        ship_type_data = {}
        for data in self.ship_data:
            ship_type = data["Vessel"]
            if ship_type not in ship_type_data:
                ship_type_data[ship_type] = {
                    "berth_waiting_times": [],
                    "mc_waiting_times": [],
                    "waiting_times": [],
                    "processing_times": [],
                    "total_times": []
                }
            ship_type_data[ship_type]["berth_waiting_times"].append(data["WT Berth occupied"])
            ship_type_data[ship_type]["mc_waiting_times"].append(data["WT meteorological constraints"])
            ship_type_data[ship_type]["waiting_times"].append(data["WT Ships in queue"])
            ship_type_data[ship_type]["processing_times"].append(data["Processing Time"])
            ship_type_data[ship_type]["total_times"].append(data["TAT"])

        ship_type_kpis = {}
        for ship_type, times in ship_type_data.items():
            ship_type_kpis[ship_type] = {
                "avg_berth_waiting_time": average_time(times["berth_waiting_times"]),
                "avg_mc_waiting_time": average_time(times["mc_waiting_times"]),
                "avg_waiting_time": average_time(times["waiting_times"]),
                "avg_processing_time": average_time(times["processing_times"]),
                "avg_total_time": average_time(times["total_times"]),
                "max_berth_waiting_time": max(times["berth_waiting_times"], default=0),
                "max_mc_waiting_time": max(times["mc_waiting_times"], default=0),
                "max_waiting_time": max(times["waiting_times"], default=0),
                "max_processing_time": max(times["processing_times"], default=0),
                "max_total_time": max(times["total_times"], default=0)
            }
        return ship_type_kpis

    def get_queue_sizes(self):
        return self.queue_sizes

    def get_forced_departures(self):
        return self.forced_departures

    def get_cargo_volume(self):
        return self.cargo_volume

    def get_total_ships_by_type(self):
        ship_type_counts = {}
        for data in self.ship_data:
            ship_type = data["Vessel"]
            if ship_type not in ship_type_counts:
                ship_type_counts[ship_type] = 0
            ship_type_counts[ship_type] += 1
        return ship_type_counts

class Ship:
    def __init__(self, env,ship_type , berth_type1,berth_type2,chanel, weather_data,start_time, mean_pre_service_time, mean_post_service_time, stats):
        self.env = env
        self.berth_type1 = berth_type1
        self.berth_type2 = berth_type2
        self.weather_data = weather_data
        self.mean_pre_service_time = mean_pre_service_time
        self.mean_post_service_time = mean_post_service_time
        self.start_time = start_time
        self.stats = stats
        self.channel = chanel
        self.ship_type = ship_type

    def ship_type0(self, ship_type=None):
        # Define the ship_types dictionary
        ship_types = {
            "category": ["A", "B", "C", "D"],
            "capacity type": [2000, 3000, 4000, 5000],
            "Cargaison": [13000, 26000, 34000, 51000],
            "Composition": [0.3, 0.5, 0.1, 0.1],
            "Taux de chargement (T/h)": [800, 1000, 1200, 1500],
            "LOA (m)": [144, 180, 196, 255]
        }

        # Convert the dictionary to a DataFrame
        df_ship_types = pd.DataFrame(ship_types)

        # Set the 'category' column as the index
        df_ship_types.set_index('category', inplace=True)
        if ship_type is None:
            ship_type = np.random.choice(["A", "B", "C", "D"], p=df_ship_types["Composition"])
        
        return ship_type,df_ship_types.loc[ship_type]
    
    def berth_type(self):
        berth_types = {
            "category": ["berth_type1", "berth_type2"],
            "LOA (m)": [190 ,250]   
        }
        
        df_berth_types = pd.DataFrame(berth_types)

        return df_berth_types

    def Meterological_accessibility(self, ship_type, current_time):
        
        # Extract weather data
        time_data  = Weather.loc[Weather['Date'] == current_time].iloc[-1]

        # Define limits for ship types
        limits = {
            "A": {"wave_height": 5, "wind_speed": 32, "wave_period": 20},
            "B": {"wave_height": 5, "wind_speed": 32, "wave_period": 20},
            "C": {"wave_height": 7, "wind_speed": 42, "wave_period": 23},
            "D": {"wave_height": 10, "wind_speed": 52, "wave_period": 27},
        }
        

        # Check weather conditions against ship limits
        if ship_type in limits:
            return (time_data['Wave height [m]']< limits[ship_type]["wave_height"]
                    and time_data['Wind speed [m/s]']< limits[ship_type]["wind_speed"]
                    and time_data['Wave period [s]'] < limits[ship_type]["wave_period"])
        
        return False    
    
    def berth_wether_limits(self, ship_type , current_time):
        current_time = int(current_time)

        # Extract weather data
        wave_height = self.weather_data['wave_height'].iloc[current_time]
        wind_speed = self.weather_data['wind_speed'].iloc[current_time]
        wave_period = self.weather_data['wave_period'].iloc[current_time]

        # Define limits for ship types
        limits = {
            "A": {"wave_height": 7, "wind_speed": 32, "wave_period": 20},
            "B": {"wave_height": 7, "wind_speed": 32, "wave_period": 20},
            "C": {"wave_height": 9, "wind_speed": 42, "wave_period": 23},
            "D": {"wave_height": 12, "wind_speed": 52, "wave_period": 27},
        }


        # Check weather conditions against ship limits
        if ship_type in limits:
            return (wave_height < limits[ship_type]["wave_height"]
                    and wind_speed < limits[ship_type]["wind_speed"]
                    and wave_period < limits[ship_type]["wave_period"])
        
        return False
    
    def berth_maintenance(self, berth_type1 ):
        if berth_type1 == "berth_type1":
            yield self.env.timeout(3)
        else:
            yield self.env.timeout(4)
            
    def processing_time(self, ship_type_data):
        
        return np.random.exponential(ship_type_data["Taux de chargement (T/h)"] / ship_type_data["Cargaison"])

    def format_time(self, sim_time):
        hours = int(sim_time)
        minutes = int((sim_time - int(sim_time)) * 60)
        return f"{hours}:{minutes:02}"
    
    def time_date(self , year , month , day , hour , minute =0):
        current_time = datetime(year, month, day, hour)
        current_time = f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d} {hour}:{minute:02d}"
        return current_time
        
    def request_channel(self , priority):
        with self.channel.request(priority) as channel_req:
            yield channel_req
            time_for_entering = np.random.exponential(0.3)
            yield self.env.timeout(time_for_entering)
            self.channel.release(channel_req)

    def process(self):
        arrival_time = self.env.now
        real_arrival_time = (self.start_time + timedelta(hours=self.env.now)).replace(year=2025)
        ship_type,ship_data = self.ship_type0(self.ship_type)
        self.stats.add_queue_size("berth_type1", len(self.berth_type1.queue))
        self.stats.add_queue_size("berth_type2", len(self.berth_type2.queue))
        self.stats.add_queue_size("berth", len(self.berth_type1.queue) + len(self.berth_type2.queue))
        
        #ship_type = self.ship_type
        current_hour = (self.start_time + timedelta(hours=self.env.now)).hour
        if current_hour >= 20 or current_hour < 8:
            # Calculate minutes until 08:00
            if current_hour >= 20:
                minutes_until_8am = (24 - current_hour + 8) * 60 - self.env.now % 60
            else:
                minutes_until_8am = (8 - current_hour) * 60 - self.env.now % 60
            yield self.env.timeout(minutes_until_8am)
        
        #berth = self.berth_type1 if ship_data["LOA (m)"] < 190 and ship_data["LOA (m)"] > 150 else self.berth_type2
        berth = self.berth_type1 if ship_type == "A"  else self.berth_type2
           
        
        with berth.request() as req:
            arrival_time = self.env.now
            yield req
            
            mc_waiting_start = self.env.now
            mc = (self.start_time + timedelta(hours=self.env.now)).replace(year=2015)
            mc_str = datetime(mc.year, mc.month, mc.day, mc.hour, mc.minute)
            #print(f"Current time: {mc_str}")
            current_hour = mc_str.hour
            current_miniute = mc_str.minute
            if current_hour % 3 == 0 and current_miniute == 0:
                next_check_hour = current_hour
            else:
                next_check_hour = (current_hour // 3 + 1) * 3

            if next_check_hour == 24:
                next_check_hour = 0
                mc = mc + timedelta(days=1)
                
            cheacking_time = datetime(mc.year, mc.month, mc.day, next_check_hour , 0)

            time_diff = cheacking_time - mc_str
            time_to_wait = time_diff.total_seconds() / 3600

            #print(f"time {mc_str} , next_check_hour {next_check_hour} , time to wait {time_to_wait}")
            Nega_time = 0
            try:
                yield self.env.timeout(time_to_wait)
            except:
                print(f'Nega time:{time_to_wait} , check time {mc_str} , next_check_hour {next_check_hour}')
                Nega_time += 1
            
            checking_time = self.time_date(mc.year, mc.month, mc.day, next_check_hour , 0)
            
                 
            while not self.Meterological_accessibility(ship_type, str(checking_time)):
                yield self.env.timeout(3)  # Check every 3 hours
            
            # Record meter
            mc_waiting_time = self.env.now - mc_waiting_start
            mc_waiting_time_r = self.format_time(mc_waiting_time)
            self.stats.add_mc_waiting_time(mc_waiting_time_r)

            # Record berth waiting time
            berth_waiting_time = self.env.now - arrival_time - mc_waiting_time
            berth_waiting_time = self.format_time(berth_waiting_time)
            self.stats.add_berth_waiting_time(berth_waiting_time)

            waiting_time = self.env.now - arrival_time
            waiting_time = self.format_time(waiting_time)
            self.stats.add_waiting_time(waiting_time)

            yield from self.request_channel(priority=1)
                
            # Processing time
            start_processing_time = self.env.now

            yield self.env.timeout(self.mean_pre_service_time)
            
            yield self.env.timeout(self.processing_time(ship_data))
            yield self.env.timeout(self.mean_post_service_time)
            
                #yield self.env.timeout(1)
            processing_time = self.env.now - start_processing_time
            processing_time = self.format_time(processing_time)
            self.stats.add_processing_time(processing_time)

            yield from self.request_channel(priority=0)

            # Total time in system
            total_time = self.env.now - arrival_time
            total_time = self.format_time(total_time)
            berth.release(req)
            self.stats.add_total_time(total_time)
            date = self.start_time + timedelta(hours=self.env.now)
            expected_tat = self.mean_pre_service_time + self.processing_time(ship_data) + self.mean_post_service_time
            departure_time = self.env.now
            self.stats.add_ship_data(real_arrival_time, ship_type, mc_waiting_time_r, waiting_time, berth_waiting_time, processing_time, total_time, expected_tat, departure_time)
            #self.stats.add_data(real_arrival_time ,ship_type, berth_waiting_time, mc_waiting_time_r, waiting_time, processing_time, total_time, date)
            #self.stats.add_ship_data(ship_type, berth_waiting_time, mc_waiting_time_r, waiting_time, processing_time, total_time, date)
def storm_event(env, process):
    while True:
        storm_duration = np.random.randint(1, 3)  # Storm duration between 1 and 3 hours
        time_until_storm = np.random.randint(5, 10)  # Random time until the next storm (in hours)
        yield env.timeout(time_until_storm)  # Wait until the storm starts
        
        print(f"Storm starts at {env.now:.2f}, ship generation interrupted.")
        process.interrupt()  # Interrupt ship generation during storm

        yield env.timeout(storm_duration)  # Storm lasts for `storm_duration` hours
        print(f"Storm ends at {env.now:.2f}, ship generation can continue.")
        
        # Allow the ship generation process to resume after the storm
        process = env.process(source.generate_ships()) 

# Parameters
mean_pre_service_time = 5  # Average pre-service time in minutes
mean_post_service_time = 5  # Average post-service time in minutes
stats = Stats()

simulation_time = 24*386  # Simulate for 7 days
start_time = datetime.strptime("00:00:00", "%H:%M:%S")

np.random.seed(46)  # For reproducibility

# Generate weather data
weather_data = generate_weather_data(simulation_time)

# Create the SimPy environment
env = simpy.Environment()
berth_type1= simpy.Resource(env, capacity=1)  # Define the mooring station resource
berth_type2= simpy.Resource(env, capacity=1)  # Define the mooring station resource
channel = simpy.PriorityResource(env, capacity=1)  # Define the channel resource

fleet_composition = pd.DataFrame({
    'Ship_Type': ['A', 'B', 'C', 'D'],
    'Capacity': [13000, 26000, 34000, 51000],
    'Composition': [0.3, 0.5, 0.1, 0.1]
})
production_rate_per_year = 3_500_000  # Example value in m³/year
initial_eta = 0  # Initial estimated time of arrival

# Create the ShipArrivalModel instance
ship_arrival_model = ShipArrivalModel(env, fleet_composition, production_rate_per_year, initial_eta)


# Create the ship generator
source = GenerateShips(env, berth_type1,berth_type2, channel, Weather,start_time, mean_pre_service_time, mean_post_service_time, stats ,ship_arrival_model)

# Start the ship generation process
env.process(source.generate_ships())

#env.process(storm_event(env, process))

# Run the simulation
env.run(until=simulation_time)

# Output the results
"""print(f"Total   {Stats.get_total_times()}")
print(f"Berth : {Stats.get_berth_waiting_times()}")
print(f"Meteor  {Stats.get_mc_waiting_times()}")
print(f"Process {Stats.get_processing_times()}")"""


"""print(f"Total ships served: {len(stats.total_times)}")
print(f"Customer average time in simulation = {average_time(stats.total_times)}")
print(f"Customer average time waiting = {average_time(stats.waiting_times)}")
print(f"Customer average time processing = {average_time(stats.processing_times)}")
print(f"Customer average time waiting for meteorological conditions = {average_time(stats.mc_waiting_times)}")
print(f"Customer average berth waiting time = {average_time(stats.berth_waiting_times)}")"""


"""current_hour = mc_str.hour
current_miniute = mc_str.minute

if current_hour % 3 == 0 and current_miniute == 0:
    next_check_hour = current_hour
else:
    next_check_hour = (current_hour // 3 + 1) * 3

if next_check_hour == 24:
    next_check_hour = 0
    mc = mc + timedelta(days=1)
    
cheacking_time = datetime(mc.year, mc.month, mc.day, next_check_hour , 0)
time_diff = cheacking_time - mc_str
time_to_wait = time_diff.total_seconds() / 3600

try:
    yield self.env.timeout(time_to_wait)
except:
    print(f'Nega time:{time_to_wait} , check time {mc_str} , next_check_hour {next_check_hour}')


checking_time = time_date(mc.year, mc.month, mc.day, next_check_hour , 0)"""