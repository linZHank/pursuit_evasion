import tensorflow as tf
import numpy as np


class DQNAgent:
    def __init__(self, name, dim_img=(150,150), dim_odom=4):
        self.name = name
        # build DQN model
        img_input = tf.keras.Input(shape=(dim_img[0],dim_img[1],3), name='img')
        odom_input = tf.keras.Input(shape=(dim_odom,), name='odom')
        img_feature = tf.keras.layers.Conv2D(16,3, padding='same', activation='relu')(img_input)
        img_feature = tf.keras.layers.MaxPool2D()(img_feature)
        img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_feature)
        img_feature = tf.keras.layers.MaxPool2D()(img_feature)
        img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(img_feature)
        img_feature = tf.keras.layers.Flatten()(img_feature)
        img_feature = tf.keras.layers.Dense(128, activation='relu')(img_feature)
        img_feature = tf.keras.layers.Dense(64, activation='relu')(img_feature)
        odom_feature = tf.keras.layers.Dense(16, activation='relu')(odom_input)
        cat_feature = tf.keras.layers.concatenate([img_feature, odom_feature])
        q_vals = tf.keras.layers.Dense(dim_act, activation='none', name='Q_values')(cat_feature)
        self.dqn_active = tf.keras.Model(inputs=[img_input, odom_input], outputs=q_vals)
        self.dqn_active.summary()
        

if __name__=='__main__':
    agent = DQNAgent(name='test_dqn_agent')
