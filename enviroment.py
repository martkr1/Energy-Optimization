from polling import TimeoutException, poll
import pandas as pd
import numpy as np


class Sim:
    "class to represent the K-spice simulator. This is the enviroment that the ML model should interact with"
    def __init__(self, timeline, app):
        self.timeline = timeline 
        self.app = app
        self.ctrl_step = 1
        self.action_space = [1, -1, 0] # Controller change is (up, down, stay)


    def import_variables(self, file):
        # Getting excel-template variables as dataframe
        df = pd.read_excel(file, sheet_name = "Sheet1", header = [0,1])

        self.df_gkpi = df["Global KPI"].dropna(how = "all", axis = 0)
        df = df.drop("Global KPI", axis = 1, level = 0)
        self.df = df.dropna(how = "all", axis = 0)
        return
    

    def check_df(self, key):

        if key in ["Input", "Ctrl", "Output"]:

            for index, row in self.df[key].iterrows():
                name = row["Block Name"]+":"+row["Variable Name"]
                try: #Jumps over blocks that does not exists and empty cells with a warning (Number of rows in df is chosen by the column with the most rows, input, output and ctrl should be equal but not global kpi.) 
                    # If there exists empty cell inbetween rows, this will ignore the values after the empty cell. #NOTE consider adding functionality to ensure this does not occur.
                    _ = self.timeline.get_value(self.app, name, row["Unit"])
                except ValueError:
                    if row["Block Name"] == "":
                        print("Only {} rows in {}".format(index, key))
                    else:
                        print("Block {} does not have a value, check if the block exists".format(row["Block Name"]))
                    break
        
        elif key == "Global KPI":

            for index, row in self.df_gkpi.iterrows():
                name = row["Block Name"]+":"+row["Variable Name"]
                try: #Jumps over blocks that does not exists and empty cells with a warning (Number of rows in df is chosen by the column with the most rows, input, output and ctrl should be equal but not global kpi.) 
                    # If there exists empty cell inbetween rows, this will ignore the values after the empty cell. #NOTE consider adding functionality to ensure this does not occur.
                    _ = self.timeline.get_value(self.app, name, row["Unit"])
                except ValueError:
                    if row["Block Name"] == "":
                        print("Only {} rows in {}".format(index, key))
                    else:
                        print("Block {} does not have a value, check if the block exists".format(row["Block Name"]))
                    break            
        else:
            print("Key is not valid")

        return


    @property 
    def state(self):
        """"Getting the current state of the simulation, represented by the values of the input variables and global KPIs provided"""

        #Inputs
        input_vals = np.empty(len(self.df["Input"]["Block Name"])) # Empty array to store values
        for index, row in self.df["Input"].iterrows():
            name = row["Block Name"]+":"+row["Variable Name"]
            input_vals[index] = self.timeline.get_value(self.app, name, row["Unit"])

        #Ctrls
        ctrl_vals = np.empty(len(self.df["Ctrl"]["Block Name"])) # Empty array to store values
        for index, row in self.df["Ctrl"].iterrows():
            name = row["Block Name"]+":"+row["Variable Name"]
            ctrl_vals[index] = self.timeline.get_value(self.app, name, row["Unit"])

        #Global KPIs
        gkpi_vals = np.empty(len(self.df_gkpi["Block Name"]))
        for index, row in self.df_gkpi.iterrows():
            name = row["Block Name"]+":"+row["Variable Name"]
            gkpi_vals[index] = self.timeline.get_value(self.app, name, row["Unit"])

        return input_vals, gkpi_vals, ctrl_vals # Return current state
    

    def get_values_polling(self, timeline, app):
        """Get input values of the state and check difference to setpoint"""


        input_vals, gkpi_vals, ctrl_vals = self.state

        a = abs(input_vals-ctrl_vals)
        
        for index, e in enumerate(a):
            if (index in self.progress_list) and (e < 0.1):
                self.progress_list.remove(index)
                print("{} has reached its setpoint {} [{}]".format(self.df["Input"]["Block Name"][index], input_vals[index], self.df["Input"]["Unit"][index]))
                print(self.progress_list)

        if not self.progress_list:
            return 0.0 # All blocks has reached its setpoint
            
        return self.timeline.achieved_speed


    def reward(self, a, b):
        """The reward should be a function of energy consumption #NOTE here we add additional criteria later production, .. etc. 
        1. The consumption goes up -> r = -1
        2. The consumtion goes down -> r = +1 """ 
        
        if np.abs(a)-np.abs(b) > 0: # (gas export - power consumption) before and after step. 
            return 1
        else: 
            return -1

    def step(self, action, terminated:bool = False, truncated:bool = False, step_change:float = 0.1):
        """Setpoint change on the provided controllers and simulation reaction"""

        #NOTE! action should array of floats - ensure this!

        a = np.sum(self.state[1]) #Energy consumption before setpoint change

        for index, row in self.df["Ctrl"].iterrows():
            var_class = self.timeline.get_variable(self.app, variable_name = row["Block Name"]+":"+row["Variable Name"])
            prev_setpoint = var_class.get_value(row["Unit"])
            setpoint = prev_setpoint + self.action_space[int(action[index].item())]*step_change
            var_class.set_value(setpoint, row["Unit"])

        self.progress_list = [i for i in range(len(self.df["Input"]["Block Name"]))]
        try:
            poll(lambda: self.get_values_polling(self.timeline, self.app) == 0.0, timeout=300, step=1) # calls func each second as timeline.achieved_speed returns > 0 and continous for max 60 seconds. i.e. this can be stopped in SimExplorer
        except TimeoutException as tee:
            print("Timeout reached")
            Truncated = True

        terminated = None #-> alarm/trip reached #NOTE not implemented

        b = np.sum(self.state[1]) #Energy consumption after setpoint change
        
        return self.state, self.reward(a, b), terminated, truncated
    

    def reset(self):
        #NOTE is there another way to do this? quite slow
        if self.timeline.achieved_speed > 0.0:
            self.timeline.pause()
        self.timeline.load_initial_condition(self.timeline.current_initial_condition) 
        self.timeline.run()  

        return self.state


    def sample(self, arg):
        if arg == "action":
            # action should be up (up, down, stay) for each controller where up and down is a change
            # with a predefined increment. 
            return np.random.choice([0, 1, 2], len(self.df["Ctrl"]["Block Name"]))