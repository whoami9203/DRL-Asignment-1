# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import pickle

def load_q_table(filename="random10_table.pkl"):
  try:
    with open(filename, "rb") as f:
      return pickle.load(f)
  except FileNotFoundError:
    print("No saved Q-table found, starting fresh!")
    return {}  # Return an empty Q-table if no file exists

obj_station_index = 0
dest_station = -1
prev_carrying = False
now_carrying = False
pick_up_once = False
passenger_pos = (-1, -1)
action = -1
state = None
epsilon = 0.5

q_table = load_q_table()

def get_action(obs):
    global obj_station_index, dest_station, prev_carrying, now_carrying, pick_up_once, passenger_pos, action
    global state, epsilon, q_table

    # print(f"action: {action}")

    def relative_dir_to_pos(my_pos, obj_pos):
      return (obj_pos[0] - my_pos[0], obj_pos[1] - my_pos[1])

    def get_state(rel_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, carrying):
      return (rel_dir[0], rel_dir[1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, carrying)
    
    taxi_row, taxi_col, x1,y1,x2,y2,x3,y3,x4,y4, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    stations = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    rel_dir = relative_dir_to_pos((taxi_row, taxi_col), stations[obj_station_index])

    # Determine whether carrying now
    if not prev_carrying and rel_dir == (0, 0) and passenger_look and action == 4: # pick up
        now_carrying = True
    elif prev_carrying and action == 5: # drop off
        passenger_pos = (taxi_row, taxi_col)
        now_carrying = False
    else:
        now_carrying = prev_carrying

    # Determine rel_dir and next_state
    if not pick_up_once and not now_carrying and rel_dir == (0, 0) and not passenger_look:
        if destination_look:
            dest_station = obj_station_index
        obj_station_index += 1
        rel_dir = relative_dir_to_pos((taxi_row, taxi_col), stations[obj_station_index])
    elif pick_up_once and not now_carrying:
        rel_dir = relative_dir_to_pos((taxi_row, taxi_col), passenger_pos)
    elif now_carrying and rel_dir == (0, 0) and not destination_look:
        if dest_station != -1:
            obj_station_index = dest_station
        else:
            obj_station_index += 1
        rel_dir = relative_dir_to_pos((taxi_row, taxi_col), stations[obj_station_index])

    if now_carrying:
        pick_up_once = True

    state = get_state(rel_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, now_carrying)
    prev_carrying = now_carrying

    if state not in q_table:
        action = np.random.choice(4)
    else:
        action = np.argmax(q_table[state])

    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

