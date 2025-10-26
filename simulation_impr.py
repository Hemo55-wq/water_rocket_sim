import numpy as np
import pandas as pd
from math import pi, sqrt
import matplotlib.pyplot as plt


class WaterRocket:
    def __init__(self, ):
        # --- Physical constants ---
        self.g = 9.81
        self.rho_air = 1.225
        self.rho_water = 1000
        self.R = 287.0
        self.gamma = 1.4
        self.T_air = 280.0
        self.T_amb = 280.0
        self.P_amb = 101325.0  # absolute atmospheric pressure

        # --- Geometry ---
        self.bottle_volume = 0.0015  # m³
        self.bottle_empty_mass = 0.15  # kg
        self.nozzle_r = 0.01
        self.nozzle_a = pi * self.nozzle_r**2
        self.bottle_a = pi * (0.05)**2
        self.Cd_nozzle = 0.8
        self.Cd_drag = 0.6

        # --- Initial conditions ---
        self.P0 = 400000.0           # absolute pressure in bottle [Pa]
        self.water_mass0 = 0.5       # kg
        self.water_mass = self.water_mass0
        self.air_volume0 = self.bottle_volume - self.water_mass0 / self.rho_water
        self.air_mass0 = (self.P0 * self.air_volume0) / (self.R * self.T_air)
        self.air_mass = self.air_mass0
        self.P = self.P0

        # --- Motion state ---
        self.height = 0.0
        self.speed = 0.0
        self.time = 0.0
        self.thrust = 0.0

        # --- Logging ---
        self.result = pd.DataFrame()

    # ----------------------------------
    # Simulation driver
    # ----------------------------------
    def simulate(self, dt=1e-4):
        # --- Water expulsion stage ---
        while self.water_mass > 0:
            self.update()
            self.step_water(dt)
        print("--- Water stage over ---")

        # --- Air expulsion (choked) ---
        while self.is_choked():
            self.update()
            self.step_air_choked(dt)
        print("--- Choked air stage over ---")
        
        # --- Air expulsion (unchoked) ---
        while self.P > self.P_amb:
            self.update()
            self.step_air_unchoked(dt)
        print("--- Unchoked air stage over ---")

        # --- Coasting phase ---
        while self.speed > 0 or self.height > 0:
            self.update()
            self.step_coast(dt)
        print("--- Rocket landed ---")

    # ----------------------------------
    # Water expulsion stage
    # ----------------------------------
    def step_water(self, dt):
        P_air = self.P
        v_e = sqrt(2 * (P_air - self.P_amb) / self.rho_water)
        m_dot = self.Cd_nozzle * self.nozzle_a * self.rho_water * v_e
        thrust = m_dot * v_e + (P_air - self.P_amb) * self.nozzle_a

        # Update air volume (due to water leaving)
        V_air_old = self.air_volume()
        self.water_mass -= m_dot * dt
        if self.water_mass < 0:
            self.water_mass = 0
        V_air_new = self.air_volume()

        # Adiabatic expansion
        self.P = self.P * (V_air_old / V_air_new) ** self.gamma

        # Motion
        net_force = thrust - self.drag_force() - self.total_mass() * self.g
        acc = net_force / self.total_mass()
        self.speed += acc * dt
        self.height += self.speed * dt
        self.time += dt
        self.thrust = thrust

    # ----------------------------------
    # Air expulsion stage (choked)
    # ----------------------------------
    def step_air_choked(self, dt):
        gamma = self.gamma
        R = self.R
        T0 = self.T_air
        P0 = self.P

        # --- Choked flow relations ---
        m_dot = (self.Cd_nozzle * self.nozzle_a * P0 *
                 np.sqrt(gamma / (R * T0) *
                         (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))))
        v_e = np.sqrt(gamma * R * T0 * (2 / (gamma + 1)))
        P_e = P0 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        T_exit = T0 * (2 / (gamma + 1))

        # --- Thrust ---
        thrust = m_dot * v_e + (P_e - self.P_amb) * self.nozzle_a

        # --- Pressure update using mass loss (adiabatic) ---
        self.air_mass -= m_dot * dt
        if self.air_mass < 0:
            self.air_mass = 0
        self.P = self.P0 * (self.air_mass / self.air_mass0) ** self.gamma

        # --- Motion integration ---
        net_force = thrust - self.drag_force() - self.total_mass() * self.g
        acc = net_force / self.total_mass()
        self.speed += acc * dt
        self.height += self.speed * dt
        self.time += dt
        self.thrust = thrust

    # ----------------------------------
    # Air expulsion stage (unchoked)
    # ----------------------------------
    def step_air_unchoked(self, dt):
        gamma = self.gamma
        R = self.R
        Cd = self.Cd_nozzle
        A_nozzle = self.nozzle_a
        P0 = self.P
        T0 = self.T_air
        P_amb = self.P_amb

        # --- Compute exit Mach number from isentropic relations ---
        # For unchoked flow, P_exit = P_amb, solve for Mach_exit iteratively
        def func(M):
            return P0 / P_amb - (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))

        # Simple Newton-Raphson or bounded iteration to estimate exit Mach
        M_e = 0.2
        for _ in range(20):
            f = func(M_e)
            df = -gamma * M_e * (1 + (gamma - 1) / 2 * M_e**2) ** (1 / (gamma - 1))
            M_e -= f / (df + 1e-9)
            M_e = max(1e-5, min(M_e, 1.0))  # constrain to subsonic regime

        # --- Exit temperature, pressure, and velocity ---
        T_e = T0 / (1 + (gamma - 1) / 2 * M_e**2)
        P_e = P_amb  # unchoked → exit pressure = ambient
        v_e = M_e * np.sqrt(gamma * R * T_e)

        # --- Mass flow rate (isentropic relation) ---
        rho_e = P_e / (R * T_e)
        m_dot = Cd * A_nozzle * rho_e * v_e

        # --- Thrust (momentum + pressure differential) ---
        thrust = m_dot * v_e + (P_e - P_amb) * A_nozzle  # second term ≈ 0

        # --- Update internal air state (mass and pressure) ---
        self.air_mass -= m_dot * dt
        if self.air_mass < 0:
            self.air_mass = 0
        self.P = self.P0 * (self.air_mass / self.air_mass0) ** self.gamma

        # --- Motion integration ---
        net_force = thrust - self.drag_force() - self.total_mass() * self.g
        acc = net_force / self.total_mass()
        self.speed += acc * dt
        self.height += self.speed * dt
        self.time += dt
        self.thrust = thrust

    # ----------------------------------
    # Coasting stage
    # ----------------------------------
    def step_coast(self, dt):
        thrust = 0.0
        net_force = -self.drag_force() - self.total_mass() * self.g
        acc = net_force / self.total_mass()
        self.speed += acc * dt
        self.height += self.speed * dt
        if self.height < 0:
            self.height = 0
        self.time += dt
        self.thrust = thrust

    # ----------------------------------
    # Helper methods
    # ----------------------------------
    def is_choked(self):
        # critical pressure ratio for dry air
        P_crit = self.P * (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1))
        return P_crit > self.P_amb

    def air_volume(self):
        return self.bottle_volume - self.water_mass / self.rho_water

    def total_mass(self):
        return self.bottle_empty_mass + self.water_mass + self.air_mass

    def drag_force(self):
        return 0.5 * self.Cd_drag * self.bottle_a * self.rho_air * self.speed**2

    def update(self):
        step = pd.DataFrame({
            "time": [self.time],
            "height": [self.height],
            "speed": [self.speed],
            "pressure_abs": [self.P],
            "water_mass": [self.water_mass],
            "air_mass": [self.air_mass],
            "thrust": [self.thrust]
        })
        self.result = pd.concat([self.result, step], ignore_index=True)

    def plot(self):
        df = self.result
        a = df.max()
        print("Peak Values \n", a)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(df.time, df.height, label="Height [m]")
        ax1.plot(df.time, df.speed, label="Speed [m/s]")
        ax1.plot(df.time, df.thrust / 10, label="Thrust [N/10]")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Values")
        ax1.legend()
        plt.title("Water Rocket Simulation (Absolute Pressure)")
        plt.show()


# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    r = WaterRocket()
    r.simulate(dt=1e-4)
    r.plot()
