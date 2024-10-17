def get_teleop_action(keyboard):
    params = load_parameters('base_parameters.yaml')

    key = float(keyboard.getKey())
    if key == 315: # up arrow
        action = np.array([0.0, params['v_max']])
    elif key == 317: # down arrow
        action = np.array([0.0, params['v_min']])
    elif key == 314: # left array
        action = np.array([params['omega_min'], 0.0]) #.00001 to not get stuck on upper bounds
    elif key == 316: # right arrow
        action = np.array([params['omega_max'], 0.0]) #.00001 to not get stuck on upper bounds
    else:
        action = np.array([0.0, 0.0])
    return action

def step(self):
    action = get_teleop_action(self.keyboard)
