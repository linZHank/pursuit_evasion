""" 
A PPO type agent class for pe env 
"""
import tensorflow as tf
import numpy as np
import logging
import tensorflow_probability as tfp
tfd = tfp.distributions


################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################

def net(dim_img, dim_odom, dim_output, activation, output_activation=None):
    """
    Take inputs of [image, odometry], output according to output_size
    Input should include two dimensions
    """
    # inputs
    img_input = tf.keras.Input(shape=(dim_img[0],dim_img[1],3), name='img')
    odom_input = tf.keras.Input(shape=(dim_odom,), name='odom')
    # image features
    img_feature = tf.keras.layers.Conv2D(16,3, padding='same', activation=activation)(img_input)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, 3, padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Conv2D(64, 3, padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D()(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(512, activation=activation)(img_feature)
    # odom features
    odom_feature = tf.keras.layers.Dense(16, activation=activation)(odom_input)
    odom_feature = tf.keras.layers.Dense(16, activation=activation)(odom_feature)
    # concatenate features
    cat_feature = tf.keras.layers.concatenate([img_feature, odom_feature])
    # outputs
    outputs = tf.keras.layers.Dense(dim_output, activation=output_activation)(cat_feature)

    return tf.keras.Model(inputs=[img_input, odom_input], outputs=outputs)

class Actor(tf.Module):
    """
    Gaussian actor
    """
    def __init__(self, dim_img, dim_odom, dim_act):
        super().__init__()
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
        self.mu_net = net(dim_img=dim_img, dim_odom=dim_odom, dim_output=dim_act, activation='relu')

    def _distribution(self, img, odom):
        mu = tf.squeeze(self.mu_net([img, odom]))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def __call__(self, img, odom, act=None):
        pi = self._distribution(img, odom)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

class Critic(tf.Module):
    def __init__(self, dim_img, dim_odom):
        super().__init__()
        self.val_net = net(dim_img=dim_img, dim_odom=dim_odom, dim_output=1, activation='relu')

    def __call__(self, img, odom):
        return tf.squeeze(self.val_net([img, odom]), axis=-1)

# class ActorCritic(tf.Module):
#     def __init__(self, dim_img, dim_odom, dim_act):
#         super().__init__()
#         self.actor = Actor(dim_img=dim_img, dim_odom=dim_odom, dim_act=dim_act)
#         self.critic = Critic(dim_img=dim_img, dim_odom=dim_odom)

class PPOAgent:
    def __init__(self, name='ppo_agent', dim_img=(150,150,3), dim_odom=4, dim_act=2, clip_ratio=0.2):
        self.clip_ratio = clip_ratio
        self.actor = Actor(dim_img, dim_odom, dim_act)
        self.critic = Critic(dim_img, dim_odom)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.actor_loss_metric = tf.keras.metrics.Mean()
        self.critic_loss_metric = tf.keras.metrics.Mean()

    def pi_given_state(self, img, odom):
        pi = self.actor._distribution(img, odom) # policy distribution (Gaussian)
        act = pi.sample()
        logp_a = self.actor._log_prob_from_distribution(pi, act)
        val = self.critic(img, odom)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, replay_buffer, num_epochs, batch_size):
        data = replay_buffer.get()
        # create actor dataset
        actor_dataset = tf.data.Dataset.from_tensor_slices(
            (data['img'],
            data['odom'],
            data['act'],
            data['adv'],
            data['logp']
        ))
        actor_dataset.shuffle(buffer_size=1024).batch(batch_size)
        # update actor
        for epch in range(num_epochs):
            logging.debug("Staring actor epoch: {}".format(epch))
            for step, batch in enumerate(actor_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.actor.trainable_variables)
                    pi, logp = self.actor(batch[0], batch[1], batch[2]) # img, odom, act
                    ratio = tf.math.exp(logp - batch[4])
                    clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), batch[3])
                    ent = tf.math.reduce_mean(pi.entropy(), axis=-1)
                    obj = tf.math.minimum(tf.math.multiply(ratio, batch[3]), clip_adv) # -.01*ent
                    loss_pi = -tf.math.reduce_mean(obj)
                # gradient descent actor weights
                grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
                self.actor_loss_metric(loss_pi)
                # log loss_pi
                if not step%100:
                    logging.debug("step {}: mean_loss = {}".format(step, self.actor_loss_metric.result()))
        # create critic dataset 
        critic_dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['odom'], data['ret']))
        critic_dataset.shuffle(buffer_size=1024).batch(batch_size)
        # update critic
        for epch in range(num_epochs):
            logging.debug("Starting critic epoch: {}".format(epch))
            for step, batch in enumerate(critic_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.critic.trainable_variables)
                    loss_v = tf.keras.losses.MSE(batch[2], self.critic(batch[0], batch[1]))
                # gradient descent critic weights
                grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
                self.critic_loss_metric(loss_v)
                # log loss_v
                if not step%100:
                    logging.debug("step {}: mean_loss = {}".format(step, self.critic_loss_metric.result()))
        
# Test agent
if __name__=='__main__':
    agent = PPOAgent()
    img = np.random.rand(10,150,150,3)
    odom = np.random.randn(10,4)
    a, v, l = agent.pi_given_state(img, odom)
