import numpy as np
#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import simpy
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
import random
from utilities import *

class ShipArrivalModel:
    def __init__(self, env, production_data ):
        """
        Initialize the ShipArrivalModel for a single product.
        
        Parameters:
        - env: SimPy environment.
        - production_data: Dictionary containing production details for a single product.
          Expected structure:
          {
              'Production per Year': <annual production>,
              'Fleet Composition': {
                  'Ship_Type_A': {'Capacity': <capacity>, 'Composition': <probability>},
                  'Ship_Type_B': {'Capacity': <capacity>, 'Composition': <probability>}
              }
          }
        """
        self.env = env
        self.production_data = production_data
        self.previous_eta = 0  # Initial ETA for the first ship
        

    def select_random_ship(self):
        """
        Randomly select a ship type based on fleet composition probabilities.
        
        Returns:
        - ship_type: Selected ship type.
        - capacity: Capacity of the selected ship type.
        """
        # Get the fleet composition dictionary
        product = list(self.production_data.keys())[0]
        fleet_composition = self.production_data[product]['Fleet Composition']
        
        # Randomly select a ship type based on probabilities
        ship_types = list(fleet_composition.keys())
        probabilities = [fleet_composition[ship]['Composition'] for ship in ship_types]
        
        # Select a ship type based on probabilities
        ship_type = np.random.choice(ship_types, p=probabilities)
        
        # Access the capacity of the selected ship type
        capacity = fleet_composition[ship_type]['Capacity']

        
        
        return ship_type, capacity , product

    def calculate_iat(self, capacity, days=366):
        """
        Calculate the Inter-Arrival Time (IAT) for the ship.
        
        Parameters:
        - capacity: Capacity of the selected ship.
        - days: Number of days in the simulation (default: 366).
        
        Returns:
        - iat: Inter-Arrival Time (in days).
        """
        # Calculate the daily production rate
        production_per_day = sum(data['Production per Year'] for data in self.production_data.values()) / days
        
        # Calculate the Inter-Arrival Time (IAT) based on ship capacity and daily production
        iat = capacity / production_per_day
        return iat



import numpy as np

class GenerateShips:
    def __init__(self, env, berths, berth_maintenance, channel, weather_data, start_time, mean_pre_service_time, 
                 mean_post_service_time, stats, production_data, storage , Night_navigation = False ):
        self.env = env
        self.berths = berths
        self.berth_maintenance = berth_maintenance
        self.weather_data = weather_data
        self.production_data = production_data
        self.mean_pre_service_time = mean_pre_service_time
        self.mean_post_service_time = mean_post_service_time
        self.start_time = start_time
        self.stats = stats
        self.channel = channel
        self.storage = storage
        self.Night_navigation = Night_navigation
        # Track inter-arrival times and arrival times per product
        self.inter_arrival_times = []
        self.ATA = []

        # Initialize inter-arrival times based on initial composition
        self._initialize_avg_iat()

    def _initialize_avg_iat(self):
        """Initialize average inter-arrival times for each product."""
        ship_arrival_model = ShipArrivalModel(self.env, self.production_data)
        
        for _ in range(1000):
            _, ship_capacity, product = ship_arrival_model.select_random_ship()
            iat = ship_arrival_model.calculate_iat(ship_capacity)
            self.inter_arrival_times.append(iat * 24)
        
        # Calculate initial average inter-arrival times
        self.avg_inter_arrival = np.mean(self.inter_arrival_times) 

    def generate_ships(self):
        """Generate ships based on dynamically recalibrated inter-arrival times."""
        ship_arrival_model = ShipArrivalModel(self.env, self.production_data)

        while True:
            ship_type, ship_capacity, product = ship_arrival_model.select_random_ship()
            avg_iat = self.avg_inter_arrival
            arrival_time = self._get_adjusted_arrival_time(avg_iat)
            
            yield self.env.timeout(arrival_time)

            # Instantiate and process the ship
            ship = self._create_ship(ship_type, product )
            self.env.process(ship.process())

            # Update remaining production and recalibrate product probabilities
            total_demande = self._update_production_and_recalibrate(ship, product)

            if total_demande < 20:
                return


            # Break condition based on minimal inter-arrival time for all products
            if self.avg_inter_arrival < 1 :
                break
            
            # Track arrival time for statistical analysis
            self.ATA.append(arrival_time)

    def _get_adjusted_arrival_time(self, avg_iat):
        """Generate an arrival time with a probabilistic adjustment to limit extreme deviations."""
        arrival_time = np.random.exponential(avg_iat)
        deviation_rate = np.abs(arrival_time - avg_iat) / avg_iat
        
        # Clamp extreme deviations to prevent outliers
        if deviation_rate > 0.8:
            arrival_time = avg_iat * (1 + 0.8 if arrival_time > avg_iat else 1 - 0.8)
        
        return arrival_time

    def _create_ship(self, ship_type, product):
        """Create a new Ship instance."""
        fleet_composition= self.production_data[product]['Fleet Composition']
        ship_data = ship_type ,fleet_composition[ship_type]
        
        return Ship(
            self.env, ship_data, product, self.berths, self.berth_maintenance, self.channel,
            self.weather_data, self.start_time, self.mean_pre_service_time, 
            self.mean_post_service_time, self.storage, self.stats , self.Night_navigation
        )

    def _update_production_and_recalibrate(self, ship, product):
        """Update production tracking and recalibrate inter-arrival times."""
        # Track production levels for each product and adjust probabilities"
        remaining_demand = self.production_data[product]['Production per Year'] / 1000 - ship.get_production(product)

        # Update probabilities based on remaining demand
        total_demand = remaining_demand

        # Clear inter-arrival time lists and recalculate averages
        self._recalculate_inter_arrival_times(remaining_demand)

        return total_demand

    def _recalculate_inter_arrival_times(self, remaining_demand):
        """Recalculate inter-arrival times based on updated remaining demand."""
        ship_arrival_model = ShipArrivalModel(self.env, self.production_data)
        
        for _ in range(100):
            _, ship_capacity, product = ship_arrival_model.select_random_ship()
            production_per_day = remaining_demand * 1000 / (366 - self.env.now / 24)
            
            #iat = ship_capacity / productions_per_day[product] if productions_per_day[product] > 0 else float('inf')
            iat = ship_capacity / production_per_day if production_per_day > 0 else float('inf')
            self.inter_arrival_times.append(iat * 24)
        
        # Update average inter-arrival times
        self.avg_inter_arrival =np.mean(self.inter_arrival_times) 

    def get_ita(self):
        return self.inter_arrival_times
    
    def get_ata(self):
        return self.ATA
    

class Berth:
    def __init__(self, env):
        self.env = env
        self.maintenance_start_date = self.generate_maintenance_date()
        self.maintenance_duration = timedelta(days=7)

    def generate_maintenance_date(self):
        """Generates a random 7-day maintenance window within the year."""
        year_start = datetime(2025, 1, 1)
        year_end = datetime(2025, 12, 31)
        # Randomly select a start date within the year, leaving room for a 7-day window
        random_day = random.randint(0, (year_end - year_start).days - 7)
        return year_start + timedelta(days=random_day)

    def is_maintenance_period(self, current_time):
        """Check if the current time is within the maintenance period."""
        maintenance_end = self.maintenance_start_date + self.maintenance_duration
        return self.maintenance_start_date <= current_time <= maintenance_end
    


class Storage:
    def __init__(self, env,loaded_amount = 0, temporary_storage=100_000 ):
        self.env = env
        self.storage = temporary_storage  
        self.loaded_amount = loaded_amount
        self.storage_capacity = temporary_storage
        self.start = 0

    def decrease_storage(self, amount):
        """Decrease the storage by a certain amount."""
        if self.storage >= amount:
            self.storage -= amount
            self.loaded_amount += amount
            return True , self.loaded_amount , 1
        elif self.storage <= amount*0.95 :
            percentage = self.storage/amount
            self.loaded_amount += self.storage
            self.storage = 0
            return False , self.loaded_amount , percentage
            
        else:
            self.storage = 0
            return False , self.loaded_amount ,0


            

    def dynamic_refill_storage(self , start_time , vessel):

        fill_rate = 4000  # 4000 m3/h

        time_spent_filling =  (self.env.now - self.start)
        theorical_time_spent_filling = (self.storage_capacity) / fill_rate
        
        self.time_to_wait = theorical_time_spent_filling - time_spent_filling
        if self.time_to_wait < 0 or self.storage == self.storage_capacity:
            self.time_to_wait = 0
        else:
            #print(f"time to wait {self.time_to_wait} storage {self.storage}     nd finally {time_spent_filling}  theorical {theorical_time_spent_filling} , time now {self.env.now} ")
            yield self.env.timeout(self.time_to_wait)   
        

        
        self.start = self.env.now
        #print(f"time to wait {self.time_to_wait} storage {self.storage}     nd finally {time_spent_filling}  theorical {theorical_time_spent_filling} , time now {self.env.now} vessel {vessel} ")
        self.storage = self.storage_capacity
        
            
       

    def actual_storage(self):
        return self.storage
            

    def refill_storage(self , amount_to_fill):
        """Simulate refilling the storage."""

        yield self.env.timeout(6)
        self.storage = self.storage_capacity # Reset to full capacity




class Stats:
    def __init__(self):
        self.total_times = []
        self.berth_waiting_times = []
        self.mc_waiting_times = []
        self.processing_times = []
        self.night_waiting_times = []
        self.waiting_times = []
        self.storage_waiting_times = []
        self.queue_sizes = {"berth_type1": [], "berth_type2": [], "berth": []}
        self.storage = []
        self.ship_data = []
        self.forced_departures = 0
        self.cargo_volume = {"exported": 0, "imported": 0}

        self.tracker = {
            'Product' : [],
            'time' : [],
            'loading' : []
        }

    def add_total_time(self, time):
        self.total_times.append(time)

    def add_berth_waiting_time(self, time):
        self.berth_waiting_times.append(time)

    def add_mc_waiting_time(self, time):
        self.mc_waiting_times.append(time)

    def add_processing_time(self, time):
        self.processing_times.append(time)

    def add_night_waiting_time(self, time):
        self.night_waiting_times.append(time)

    def add_waiting_time(self, time):
        self.waiting_times.append(time)

    def add_storage_waiting_time(self, time):
        self.storage_waiting_times.append(time)

    def add_queue_size(self, queue_type, size):
        self.queue_sizes[queue_type].append(size)
    
    def add_storage(self, storage):
        self.storage.append(storage)

    def add_ship_data(self, arrival, vessel_ID, vessel, Product_type, Berth_claimed, Finished_loading, Berth_released, wt_meteorological_constraints, wt_ships_in_queue, wt_berth_occupied, wt_night, processing_time,berth_occupied, tat, expected_tat, departure_time, cargo_loaded):
        self.ship_data.append({
            "Arrival": arrival,
            "vessel_id": vessel_ID,
            "Vessel": vessel,
            "Product_name": Product_type,
            "Berth_claimed": Berth_claimed,
            "Finished_loading": Finished_loading,
            "Berth_released": Berth_released,
            "WT meteorological constraints": wt_meteorological_constraints,
            "WT Ships in queue": wt_ships_in_queue,
            "WT Berth occupied": wt_berth_occupied,
            "WT Night": wt_night,
            "Port Operation": processing_time,
            "berth occupied": berth_occupied,
            "TAT": tat,
            "Expected TAT": expected_tat,
            "Departure_Time": departure_time,
            "cargo_loaded": cargo_loaded
        })

    def increment_forced_departures(self):
        self.forced_departures += 1

    def add_cargo_volume(self, volume, direction):
        if direction == "exported":
            self.cargo_volume["exported"] += volume
        elif direction == "imported":
            self.cargo_volume["imported"] += volume

    def get_max_kpis(self):
        max_berth_waiting_time = max_time(self.berth_waiting_times)
        max_mc_waiting_time = max_time(self.mc_waiting_times)
        max_waiting_time = max_time(self.waiting_times)
        max_processing_time = max_time(self.processing_times)
        max_total_time = max_time(self.total_times)
        return {
            "max_berth_waiting_time": max_berth_waiting_time,
            "max_mc_waiting_time": max_mc_waiting_time,
            "max_waiting_time": max_waiting_time,
            "max_Port Operation": max_processing_time,
            "max_total_time": max_total_time
        }

    def get_average_kpis(self):
        avg_berth_waiting_time = average_time(self.berth_waiting_times)
        avg_mc_waiting_time = average_time(self.mc_waiting_times)
        avg_waiting_time = average_time(self.waiting_times)
        avg_storage_waiting_time = average_time(self.storage_waiting_times)
        avg_processing_time = average_time(self.processing_times)
        avg_total_time = average_time(self.total_times)

        return {
            "avg_berth_waiting_time": avg_berth_waiting_time,
            "avg_mc_waiting_time": avg_mc_waiting_time,
            "avg_waiting_time": avg_waiting_time,
            "avg_storage_waiting_time": avg_storage_waiting_time,
            "avg_Port Operation": avg_processing_time,
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
                    "Port Operation": [],
                    "total_times": []
                }
            ship_type_data[ship_type]["berth_waiting_times"].append(data["WT Berth occupied"])
            ship_type_data[ship_type]["mc_waiting_times"].append(data["WT meteorological constraints"])
            ship_type_data[ship_type]["waiting_times"].append(data["WT Ships in queue"])
            ship_type_data[ship_type]["Port Operation"].append(data["Port Operation"])
            ship_type_data[ship_type]["total_times"].append(data["TAT"])

        ship_type_kpis = {}
        for ship_type, times in ship_type_data.items():
            ship_type_kpis[ship_type] = {
                "avg_berth_waiting_time": average_time(times["berth_waiting_times"]),
                "avg_mc_waiting_time": average_time(times["mc_waiting_times"]),
                "avg_waiting_time": average_time(times["waiting_times"]),
                "avg_Port Operation": average_time(times["Port Operation"]),
                "avg_total_time": average_time(times["total_times"]),
                "max_berth_waiting_time": max_time(times["berth_waiting_times"]),
                "max_mc_waiting_time": max_time(times["mc_waiting_times"]),
                "max_waiting_time": max_time(times["waiting_times"]),
                "max_Port Operation": max_time(times["Port Operation"]),
                "max_total_time": max_time(times["total_times"])
            }
        return ship_type_kpis

    def get_queue_sizes(self):
        return self.queue_sizes

    def get_forced_departures(self):
        return self.forced_departures

    def get_cargo_volume(self):
        return self.cargo_volume
    
    def get_storage(self):
        return self.storage

    def get_total_ships_by_type(self):
        ship_type_counts = {}
        for data in self.ship_data:
            ship_type = data["Vessel"]
            if ship_type not in ship_type_counts:
                ship_type_counts[ship_type] = 0
            ship_type_counts[ship_type] += 1
        return ship_type_counts
    
    def get_tracker(self):
        return self.tracker 

class Ship:
    def __init__(self, env,ship_data,product , berths ,berth_maintenance,chanel, weather_data,start_time
                 , mean_pre_service_time, mean_post_service_time, storage , stats, Night_navigation  ):
        self.env = env
        self.berths = berths
        self.berth_c = berth_maintenance
        self.weather_data = weather_data
        self.mean_pre_service_time = mean_pre_service_time
        self.mean_post_service_time = mean_post_service_time
        self.start_time = start_time
        self.stats = stats
        self.channel = chanel
        self.ship_data = ship_data
        self.storage = storage
        self.product = product
        self.Night_navigation = Night_navigation
        self.temporary_storage = self.storage.storage_capacity
        self.loading_tracker = self.stats.tracker

    def Meterological_accessibility(self, ship_type, current_time):
        
        # Extract weather data
        time_data  = self.weather_data.loc[self.weather_data['Date'] == current_time].iloc[-1]

        # Define limits for ship types
        limits = {
            "A": {"wave_height": 2, "wind_speed": 10},
            "B": {"wave_height": 2, "wind_speed": 10},
            "C": {"wave_height": 2, "wind_speed": 10},
            "D": {"wave_height": 2, "wind_speed": 10},
            "E": {"wave_height": 2, "wind_speed": 10}
        }
        

        # Check weather conditions against ship limits
        if ship_type in limits:
            return (time_data['Wave height [m]']< limits[ship_type]["wave_height"]
                    and time_data['Wind speed [m/s]']< limits[ship_type]["wind_speed"])
        
        return False     
    
    def loading_condition_met(self, ship_type, current_time):
        # Extract weather data
        time_data = self.weather_data.loc[self.weather_data['Date'] == current_time].iloc[-1]
        
        # Define limits for ship types
        limits = {
            "A": {"wave_height": 1.5, "wind_speed": 22},
            "B": {"wave_height": 1.5, "wind_speed": 22},
            "C": {"wave_height": 1.5, "wind_speed": 22},
            "D": {"wave_height": 2, "wind_speed": 22},
            "E": {"wave_height": 2, "wind_speed": 22}
        }

        # Check weather conditions against ship limits
        if ship_type in limits:
            return (time_data['Wave height [m]'] < limits[ship_type]["wave_height"]
                    and time_data['Wind speed [m/s]'] < limits[ship_type]["wind_speed"])  
        return False
    
    def forced_departure(self, ship_type, current_time):
        # Extract weather data
        time_data = self.weather_data.loc[self.weather_data['Date'] == current_time].iloc[-1]

        # Define limits for ship types
        limits = {
            "A": {"wave_height": 2, "wind_speed": 25},
            "B": {"wave_height": 2, "wind_speed": 25},
            "C": {"wave_height": 2, "wind_speed": 25},
            "D": {"wave_height": 2, "wind_speed": 25}
        }

        # Check weather conditions against ship limits
        if ship_type in limits:
            return (time_data['Wave height [m]'] > limits[ship_type]["wave_height"]
                    or time_data['Wind speed [m/s]'] > limits[ship_type]["wind_speed"])
        return False
  
    def filling_rate(self, time_spent_loading, ship_type_data):
        return ship_type_data["Loading Rate (T/h)"] * time_spent_loading
        
    def wait_for_better_weather(self, env , start_time , ship_type ):

        current_time = (start_time + timedelta(hours=env.now)).replace(year=2025)
        closest_time = get_closest_time(current_time)
        closest_time = time_date(2015 ,closest_time.month, closest_time.day, closest_time.hour, 0)

        while not self.Meterological_accessibility(ship_type, str(closest_time)):
            current_time = (start_time + timedelta(hours=env.now)).replace(year=2025)
            closest_time = get_closest_time(current_time)
            closest_time = time_date(2015 ,closest_time.month, closest_time.day, closest_time.hour, 0)
            #print(f"waiting for better weather conditions at {closest_time}")
            yield env.timeout(1.5)  # Check every 3 hours
        
    def loading_time(self, ship_type_data):
        
        return ship_type_data["Capacity"]/ship_type_data["Loading Rate (T/h)"] 
        
    def request_channel(self , priority):
        with self.channel.request(priority) as channel_req:
            yield channel_req
            time_for_entering = np.random.exponential(1.5)
            yield self.env.timeout(time_for_entering)
            self.channel.release(channel_req)

    def record_time(self ,env, start_time):

        time = env.now - start_time
        formatted_time = format_time(time)
        return formatted_time

    def handle_night_navigation(self):
        current_hour = (self.start_time + timedelta(hours=self.env.now)).hour
        if not self.Night_navigation and (current_hour >= 22 or current_hour < 8):
            hours_until_8am = (8 - current_hour) % 24
            yield self.env.timeout(hours_until_8am)  # Wait until 8 AM  

    def ship_loading(self , ship_type , ship_data , time_now, time_step , loading_time , loading_rate_factor):
        load = 0
        added_time = 0
    
        while self.env.now < time_now + loading_time + added_time and load < ship_data["Capacity"]:
            
                
            current_time = (self.start_time + timedelta(hours=self.env.now)).replace(year=2025)
            closest_time = get_closest_time(current_time)
            closest_time = time_date(2015 ,closest_time.month, closest_time.day, closest_time.hour, 0)

            if self.loading_condition_met(ship_type, str(closest_time)): 
                yield self.env.timeout(time_step)
                amount_to_load = self.filling_rate(time_step, ship_data) * loading_rate_factor
                load += amount_to_load
                Tr , loaded_amount ,_ = self.storage.decrease_storage(amount_to_load)
                self.percentage = (load / ship_data["Capacity"] )* 100

                self.loading_tracker['Product'].append(self.product)
                self.loading_tracker['time'].append(self.env.now)
                self.loading_tracker['loading'].append(amount_to_load)

                """if (not Tr) :
                    # Refill the storage
                    start_waiting_storage = self.env.now
                    
                    yield self.env.process(self.storage.dynamic_refill_storage(self.storage.start , self.vessel_id)) 
                    self.storage_waiting_time += self.env.now - start_waiting_storage
                    added_time += self.storage.time_to_wait"""

                if (not Tr) :
                    # Refill the storage
                    start_waiting_storage = self.env.now
                    yield self.env.process(self.storage.refill_storage(self.storage.storage_capacity)) 
                    self.storage_waiting_time += self.env.now - start_waiting_storage
                    
                    added_time += 6 #self.storage.storage_capacity / 8000
                    yield self.env.timeout(time_step) 
                    Tr , loaded_amount ,_= self.storage.decrease_storage(amount_to_load)
                    load += amount_to_load
                    self.percentage = (load / ship_data["Capacity"] )* 100
                
            elif self.forced_departure(ship_type, str(closest_time)) :
                self.stats.increment_forced_departures()
                if self.percentage > 0.95:
                    break
                else:
                    return
                        
            else:
                #print(f"vessel_id {vessel_id} needs to wait for better weather conditions")
                yield self.env.timeout(2)
                added_time += 2

    def process(self):
        

        # Record the arrival time
        arrival_time = self.env.now
        arrival_time0 = arrival_time
        real_arrival_time = (self.start_time + timedelta(hours=self.env.now)).replace(year=2025)
        real_arrival_time = datetime(real_arrival_time.year ,real_arrival_time.month, real_arrival_time.day, real_arrival_time.hour, real_arrival_time.minute)

        avilaible_berth = [i for i,berth in enumerate(self.berths) if berth.count == 0]
        queue_sizes = [len(berth.queue) for berth in self.berths]
        i = np.random.choice(avilaible_berth) if len(avilaible_berth) > 0 else np.argmin(queue_sizes)
        berth = self.berths[i]
        

        # Generate ship data
        Product_type = self.product
        ship_type,ship_data = self.ship_data
        queue_sizes = [len(berth.queue) for berth in self.berths]
        #self.stats.add_queue_size("berth_type1", sum(queue_sizes))
        vessel_id = "Vessel"+ ship_type +str(np.random.randint(0,1000))
        self.vessel_id = vessel_id  

        
        # Check if the ship arrives during the night
        current_hour = (self.start_time + timedelta(hours=self.env.now)).hour

        if self.Night_navigation == False :
            if current_hour >= 22 or current_hour < 8:
                # Calculate the time until 8 AM the next day
                hours_until_8am = (8 - current_hour) % 24
                yield self.env.timeout(hours_until_8am)
            
        # Record night waiting time
        night_waiting_time_h = self.env.now - arrival_time
        night_waiting_time = format_time(night_waiting_time_h)
        self.stats.add_night_waiting_time(night_waiting_time)
         
        
        with berth.request() as req:
            arrival_time = self.env.now
            req.ship_name = vessel_id
            
            queue_sizes = [len(berth.queue) for berth in self.berths]
            self.stats.add_queue_size("berth_type1", sum(queue_sizes))
            #print(f"queue {len(berth.queue)}")
            # Request the berth
            yield req
            berth_waiting_time = self.env.now - arrival_time
            mc_waiting_start = self.env.now
            
            # Check if the ship can enter the port
            yield from self.wait_for_better_weather(self.env , self.start_time , ship_type)

            # Record meter
            mc_waiting_time = self.env.now - mc_waiting_start
            mc_waiting_time_r = self.record_time(self.env ,mc_waiting_start )
            self.stats.add_mc_waiting_time(mc_waiting_time_r)

            # Record berth waiting time
            #berth_waiting_time = self.env.now - arrival_time - mc_waiting_time
            berth_waiting_time = format_time(berth_waiting_time)
            self.stats.add_berth_waiting_time(berth_waiting_time)

            # Record waiting time
            waiting_time_h = self.env.now - arrival_time
            waiting_time = format_time(waiting_time_h)
            self.stats.add_waiting_time(waiting_time)

            # Claim the berth
            berth_claimed = (real_arrival_time + timedelta(hours=waiting_time_h+night_waiting_time_h))
            berth_claimed = datetime(berth_claimed.year ,berth_claimed.month, berth_claimed.day, berth_claimed.hour, berth_claimed.minute)

            # Request channel
            yield from self.request_channel(priority=1)
              
            # Processing time
            start_processing_time = self.env.now
            
    
            pre_service_time = np.random.exponential(self.mean_pre_service_time)
            post_service_time =  np.random.exponential(self.mean_post_service_time)

            yield self.env.timeout(1)
            # Processing time
            start_processing_time = self.env.now
            yield self.env.timeout(pre_service_time)
            current_time = (self.start_time + timedelta(hours=self.env.now)).replace(year=2025)

            if self.berth_c.is_maintenance_period(current_time):   
                loading_rate_factor = 2/3
            else:
                loading_rate_factor = 1

            loading_time = self.loading_time(ship_data) * (1/loading_rate_factor)
            time_now = self.env.now
            time_step = loading_time/30
            self.storage_waiting_time = 0
            Storage = self.storage.actual_storage()
            self.stats.add_storage(Storage)

            if Storage == self.temporary_storage:
                self.storage.start = self.env.now
            
            
            
            yield self.env.process(self.ship_loading(ship_type , ship_data , time_now, time_step , loading_time , loading_rate_factor))

            # Record storage waiting time
            self.storage_waiting_time = format_time(self.storage_waiting_time)
            self.stats.add_storage_waiting_time(self.storage_waiting_time)

            cargo_loaded = (ship_data["Capacity"]*(self.percentage/100))/1000
            # Record finish loading time
            finish_loading = (real_arrival_time + timedelta(hours=night_waiting_time_h +self.env.now - arrival_time))
            finish_loading = datetime(finish_loading.year ,finish_loading.month, finish_loading.day, finish_loading.hour, finish_loading.minute)
            
            yield self.env.timeout(post_service_time)
            # Record processing time
            processing_time0 = self.env.now - start_processing_time
            processing_time = format_time(processing_time0)
            self.stats.add_processing_time(processing_time)
            yield self.env.timeout(1)
            
            # Request channel
            yield from self.request_channel(priority=0)

            # Record total time in system (TAT)
            total_time = self.env.now - arrival_time0
            total_time = format_time(total_time)
            self.stats.add_total_time(total_time)

            #Release the berth and record the time
            berth.release(req)
            berth_release = (real_arrival_time + timedelta(hours=night_waiting_time_h +self.env.now - arrival_time))
            berth_release = datetime(berth_release.year , berth_release.month, berth_release.day, berth_release.hour, berth_release.minute)
            
            # Calculate the expected TAT and departure time
            departure_time = self.start_time + timedelta(hours=self.env.now)
            departure_time = datetime(departure_time.year , departure_time.month, departure_time.day, departure_time.hour, departure_time.minute)
            expected_tat = loading_time + pre_service_time + post_service_time
            
            expected_tat = format_time(expected_tat)
            self.stats.add_ship_data(
                real_arrival_time,
                vessel_id,  # Example vessel ID
                ship_type,
                Product_type, 
                berth_claimed,
                finish_loading,
                berth_release,
                mc_waiting_time_r,
                waiting_time,
                berth_waiting_time,
                night_waiting_time,
                processing_time,
                Product_type+" "+str(i),
                total_time,
                expected_tat,
                departure_time.replace(year=2025),
                cargo_loaded
            )

    def get_production(self, product):
        loaded_amount = sum(ship['cargo_loaded'] for ship in self.stats.ship_data if ship['Product_name'] == product)
        #print(f"product {product} , loaded_amount {loaded_amount}")
        return loaded_amount

   
