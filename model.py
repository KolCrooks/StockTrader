import tensorflow as tf
import tensorflow.keras as keras

import settings
import dataManager

import numpy as np
import gym
import time

from termcolor import colored, cprint

def createModel(state_space_len: int, action_space_len: int):
    try:
        model = keras.models.load_model('./saves/model.h5')
    except:
        model = keras.models.Sequential([
            keras.layers.GRU(settings.GRU_UNITS, input_shape=(None, state_space_len)),
            keras.layers.Dense(settings.DENSE_UNITS, activation="relu"),
            keras.layers.Dropout(settings.DROPOUT),
            keras.layers.Dense(action_space_len, activation="softmax")
        ], name="Stock_Trader")
        model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    model.summary()
    return model
 
class TradingModel:
    model: keras.Model

    def __init__(self):
        self.model = createModel(dataManager.N_DISCRETE_STATES, dataManager.N_DISCRETE_ACTIONS)
        
    def QLearn(self, env: gym.Env, action_cnt: int, symbol: str):
        r_avg_list = []

        eps = settings.EPS

        r_sum = 0

        for i in range(action_cnt):
            eps *= settings.DECAY_FACTOR
            
            state = env.state()

            if np.random.random() < eps:
                a = np.random.randint(0, dataManager.N_DISCRETE_ACTIONS)
            else:
                a = np.argmax(self.model.predict(np.reshape(state, (1, 1, -1))))

            new_state, reward, done, _ = env.step(a)
            if done:
                return
            target = reward + settings.Y_VAL * np.max(self.model.predict(np.reshape(new_state, (1, 1, -1))))
            target_vec = self.model.predict(np.reshape(state, (1, 1, -1)))[0] # get the first prediction epoch
            target_vec[a] = target # change val of target action to be action with greatest reward
            self.model.fit(np.reshape(state, (1, 1, -1)), target_vec.reshape(-1, dataManager.N_DISCRETE_ACTIONS), epochs=1)
            print("Action {} of {} on state: {}".format(colored(i, 'blue'), colored(action_cnt, 'blue'), colored(str(state), 'cyan')))
            print("{} - EPS: {}".format(colored(dataManager.actions[a], 'red'), colored(eps, 'magenta')))
            env.render()
            time.sleep(settings.TRADE_INTERVAL)
            if i % 60:
                self.model.reset_metrics()
                self.model.save('./saves/model.h5')
            r_sum += reward
