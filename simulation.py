import numpy as np 
import pandas as pd 
from math import pi, sqrt
import matplotlib.pyplot as plt


class water_rocket: 
    def __init__(self): 
        #Paramaters 
        self.init_pressure = 400000             # pascal absolute in bottle 
        self.ambient_pressure = 101325          # atmospheric pressure used for choked / unchoked flow 
        self.init_water = 0.5                   # water in kg 

        self.bottle_volume = 0.0015             # volume in m3
        self.bottle_empty_weight = 0.15         # weight of the empty Rocket 
        
        #Aerodynamics
        self.bottle_cd = 1                      # coefficient of drag 
        self.bottle_a = pi * pow(0.05, 2)       # area of bottle 
        self.nozzle_a = pi * pow(0.005, 2)      # area of nozzle 
        
        #Assumptions 
        self.discharge_coefficient = 1          # ideal nozzle without resistance or losses 
        self.water_density = 1000               # density of water 
        self.air_gamma = 1.4                    # ratio of specific heats for air
        self.air_R = 287.0                      # gas constant J/(kg·K)     
        self.air_temp = 280.0        
        self.ambient_temp = 280.0   
        
        self.speed = 0 
        self.init_weight = self.bottle_empty_weight + self.init_water
        self.water = self.init_water
        self.pressure = self.init_pressure
        self.height = 0
        self.time = 0
        self.thrust = 0
        
        self.result = pd.DataFrame()
        
    def simulate(self, dt): 
        while(self.water > 0): 
            self.update()
            self.step_water(0.0001)
        print("---Water Stage Over ---")
        self.water = 0
        
        while(self.choked()): 
            self.update()
            self.step_air_choked(0.01)
        print("---Choked Air Stage Over ---")

        while(self.speed > 0): 
            self.update()
            self.step_coast(0.001)
        print("---Rocket landed (crashed) ---")
        print(self.result)
        
        return

        
        while(self.pressure > 0): 
            self.update()
            self.step_air_unchoked(0.01)
        print("---Unchoked Air Stage Over ---")
         
        while(self.height > 0): 
            self.update()
            self.step_coast(0.05)
        print("---Rocket landed (crashed) ---")
        
    def step_water(self, dt): 
        mass_exchange_rate = self.discharge_coefficient * (self.nozzle_a * sqrt(2 * self.water_density * (self.pressure)))
        nozzle_velocity = self.discharge_coefficient * sqrt((2 * self.pressure) / self.water_density)        
        
        self.thrust = nozzle_velocity * mass_exchange_rate - (self.air_resistance() + self.gravitation())
        
        mass_exchange = mass_exchange_rate * dt 
        P0_before = self.pressure + self.ambient_pressure
        self.pressure *= (self.air_volume() /  (self.air_volume() + (mass_exchange/1000))) ** self.air_gamma    # adiabatic air expansion 
        self.air_temp *= (self.pressure + self.ambient_pressure) / P0_before ** ((self.air_gamma- 1) / self.air_gamma)
        self.speed += (self.thrust / self.weight()) * dt
        self.height += self.speed * dt 
        self.water -= mass_exchange
        self.time += dt
       
    def step_air_choked(self, dt):              # using dry air for mass calculation (Room for improvement )
        gamma = self.air_gamma   
        mass_exchange_rate = (self.bottle_cd * self.nozzle_a * 
                              (self.pressure + self.ambient_pressure) * 
                              np.sqrt(gamma / (self.air_R * self.air_temp) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))))
        v_e = np.sqrt(gamma * self.air_R * self.air_temp * (2 / (gamma + 1)))
        
        thrust = mass_exchange_rate * v_e + ((self.pressure + self.ambient_pressure) * (2 / (gamma + 1)) ** (gamma / (gamma - 1)) - self.ambient_pressure) * self.nozzle_a
        self.pressure 
        dPdt = - gamma * self.air_R * T_exit / self.air_volume() * mass_exchange_rate

        # Update pressure
        self.pressure += dPdt * dt

        
        #self.air_temp *= (self.pressure + self.ambient_pressure) / P0_before ** ((self.air_gamma- 1) / self.air_gamma)

    def step_air_unchoked(self, dt): 
        pass 
    
    def step_coast(self, dt): 
        self.thrust = - self.air_resistance() - self.gravitation()
        
        self.speed += (self.thrust / self.weight()) * dt
        self.height += self.speed * dt 
        self.time += dt
        
    def choked(self): 
        return((self.ambient_pressure / (self.pressure + self.ambient_pressure)) < 0.455)            # choke ratio for humid air 
    
    def weight(self): 
        return self.bottle_empty_weight + self.water
    
    def air_volume(self): 
        return (self.bottle_volume - (self.water/1000))             # Bottle volume minus water volume ( in liters )
    
    def air_resistance(self): 
        return ( 0.5 * (self.speed ** 2) * self.bottle_a * self.bottle_cd * 1.225) 
    
    def gravitation(self):              # returns gravitational force on rocket (as a positive force)
        return self.weight() * 9.81     

    def update(self): 
        step = pd.DataFrame({"time": [self.time], 
                             "height": [self.height], 
                             "water": [self.water], 
                             "pressure": [self.pressure], 
                             "speed": [self.speed],
                             "thrust": [self.thrust],
                             "air_resistance": [self.air_resistance()],
                             "gravitation": [self.gravitation()],
                             "weight": [self.weight()]})
        self.result = pd.concat([self.result, step])
    
    def plot(self): 
        df = self.result.iloc[2:] 
        plt.plot(df.time, df.air_resistance, label='Air_resistance')
        plt.plot(df.time, df.speed, label='Speed')
        plt.plot(df.time, df.height, label='Height')
        plt.plot(df.time, df.gravitation, label='gravity')
        plt.legend()
        plt.show()

r = water_rocket()
r.simulate(0.01)
r.plot()





