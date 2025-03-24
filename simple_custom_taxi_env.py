# import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, grid_size_min=5, grid_size_max=10, fuel_limit=1000, fix_size=True, corner_station=False, full_obstacle=False):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.corner_station = corner_station
        self.grid_size_min = grid_size_min
        self.grid_size_max = grid_size_max
        self.grid_size = grid_size_min
        self.fix_size = fix_size
        self.full_obstacle = full_obstacle
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        self.stations = [(0, 0), (self.grid_size - 1, self.grid_size - 1), (0, self.grid_size - 1), (self.grid_size - 1, 0)]
        self.passenger_loc = None

        self.obstacles = set()  # No obstacles in simple version
        self.destination = None

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        if not self.fix_size:
          self.grid_size = random.randint(self.grid_size_min, self.grid_size_max)
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False


        # 2. Generate stations: choose 4 positions such that none are adjacent.
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        random.shuffle(all_cells)
        if not self.corner_station:
          stations = []
          for cell in all_cells:
              valid = True
              for s in stations:
                  # Check 8-neighborhood (or use 4-neighborhood if desired)
                  if abs(cell[0] - s[0]) + abs(cell[1] - s[1]) <= 1:
                      valid = False
                      break
              if valid:
                  stations.append(cell)
                  if len(stations) == 4:
                      break
          # If for some reason we cannot find 4, fallback to fixed corners.
          if len(stations) < 4:
              stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
          self.stations = stations

        # 3. Generate obstacles.
        # Maximum obstacles: under 10% of grid cells.
        total_cells = self.grid_size * self.grid_size
        if self.full_obstacle:
          max_obstacles = int(0.09 * total_cells)
        else:
          max_obstacles = random.randint(2, int(0.09 * total_cells))
        obstacles = set()
        available_cells = set(all_cells) - set(self.stations)
        candidate_list = list(available_cells)
        random.shuffle(candidate_list)
        for cell in candidate_list:
            if len(obstacles) >= max_obstacles:
                break
            # Tentatively add an obstacle.
            obstacles.add(cell)
            # Check connectivity among stations.
            # if not self._check_connectivity(obstacles):
            #     obstacles.remove(cell)
        self.obstacles = obstacles

        # 4. Set taxi starting position: choose from cells that are not stations or obstacles.
        available_positions = [cell for cell in all_cells if cell not in self.stations and cell not in self.obstacles]
        self.taxi_pos = random.choice(available_positions)

        # 5. Set passenger and destination.
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1


        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                else:
                    reward = -10
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=10

        reward -= 0.1

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}



        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle


        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''


        labels = ['R', 'G', 'Y', 'B']
        for i, station in enumerate(self.stations):
            x, y = station
            grid[x][y] = labels[i]

        # Mark obstacles with 'X'
        for (x, y) in self.obstacles:
            grid[x][y] = 'X'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 50,
        "grid_size_min": 5,
        "grid_size_max": 10,
        "fix_size": False,
        "corner_station": False,
        "full_obstacle": False
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")