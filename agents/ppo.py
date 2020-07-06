""" 
A PPO type agent class for pe env 
"""
import tensorflow as tf
import numpy as np
import logging
import tensorflow_probability as tfp
tfd = tfp.distributions
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


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

def net(dim_inputs, dim_outputs, activation, output_activation=None):
    """
    Take inputs of [image, odometry], output according to output_size
    Input should include two dimensions
    """
    # inputs
    img_inputs = tf.keras.Input(shape=dim_inputs, name='img_inputs')
    # image features
    img_feature = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=activation)(img_inputs)
    img_feature = tf.keras.layers.MaxPool2D((2,2))(img_feature)
    img_feature = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D((2,2))(img_feature)
    img_feature = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(64, activation=activation)(img_feature)
    # outputs
    outputs = tf.keras.layers.Dense(dim_outputs, activation=output_activation)(img_feature)

    return tf.keras.Model(inputs=img_inputs, outputs=outputs)

class Actor(tf.Module):
    def __init__(self, dim_img, dim_act):
        super().__init__()
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
        self.mu_net = net(dim_inputs=dim_img, dim_outputs=dim_act, activation='relu')

    def _distribution(self, img):
        mu = tf.squeeze(self.mu_net(img))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def __call__(self, img, act=None):
        pi = self._distribution(img)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a

class Critic(tf.Module):
    def __init__(self, dim_img):
        super().__init__()
        self.val_net = net(dim_inputs=dim_img, dim_outputs=1, activation='relu')

    def __call__(self, img):
        return tf.squeeze(self.val_net(img), axis=-1)


class PPOAgent:
    def __init__(self, name='ppo_agent', dim_img=(80,80,3), dim_act=2, clip_ratio=0.2, lr_actor=3e-4,
                 lr_critic=1e-3, batch_size=32, target_kl=0.01):
        self.clip_ratio = clip_ratio
        self.actor = Actor(dim_img, dim_act)
        self.critic = Critic(dim_img)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.actor_loss_metric = tf.keras.metrics.Mean()
        self.critic_loss_metric = tf.keras.metrics.Mean()
        self.batch_size = batch_size
        self.target_kl = target_kl

    def pi_of_a_given_s(self, img):
        pi = self.actor._distribution(img) # policy distribution (Gaussian)
        act = pi.sample()
        logp_a = self.actor._log_prob_from_distribution(pi, act)
        val = tf.squeeze(self.critic(img), axis=-1)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, actor_dataset, critic_dataset, num_epochs):
        # update actor
        batched_actor_dataset = actor_dataset.shuffle(1024).batch(self.batch_size)
        for epch in range(num_epochs):
            logging.debug("Staring actor epoch: {}".format(epch+1))
            ep_kl = tf.convert_to_tensor([]) 
            ep_ent = tf.convert_to_tensor([]) 
            for step, batch in enumerate(batched_actor_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.actor.trainable_variables)
                    pi, logp = self.actor(batch['img'], batch['act']) # img, odom, act
                    ratio = tf.math.exp(logp - batch['logp']) # pi/old_pi
                    clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio),
                                                batch['adv'])
                    obj = tf.math.minimum(tf.math.multiply(ratio, batch['adv']), clip_adv) # -.01*ent
                    loss_pi = -tf.math.reduce_mean(obj)
                    approx_kl = batch['logp'] - logp
                    ent = tf.math.reduce_mean(pi.entropy(), axis=-1)
                # gradient descent actor weights
                grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
                self.actor_loss_metric(loss_pi)
                # record kl-divergence and entropy
                ep_kl = tf.concat([ep_kl, approx_kl], axis=0)
                ep_ent = tf.concat([ep_ent, ent], axis=0)
                # log loss_pi
                if not step%100:
                    logging.debug("pi update step {}: mean_loss = {}".format(step, self.actor_loss_metric.result()))
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            logging.info("Epoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                epch+1,
                self.actor_loss_metric.result(),
                entropy,
                kl
            ))
            # early cutoff due to large kl-divergence
            if kl > 1.5*self.target_kl:
                logging.warning("Early stopping at epoch {} due to reaching max kl-divergence.".format(epch+1))
                break
        # update critic
        batched_critic_dataset = critic_dataset.shuffle(1024).batch(self.batch_size)
        for epch in range(num_epochs):
            logging.debug("Starting critic epoch: {}".format(epch))
            for step, batch in enumerate(batched_critic_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.critic.trainable_variables)
                    loss_v = tf.keras.losses.MSE(batch['ret'], self.critic(batch['img']))
                # gradient descent critic weights
                grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
                self.critic_loss_metric(loss_v)
                # log loss_v
                if not step%100:
                    logging.debug("v update step {}: mean_loss = {}".format(step, self.critic_loss_metric.result()))
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            logging.info("Epoch :{} \nLoss: {}".format(
                epch+1,
                self.critic_loss_metric.result()
            ))
        
# Test agent
if __name__=='__main__':
    agent = PPOAgent()
    # test dataset
    img = np.random.rand(100,80,80,3).astype(np.float32)
    odom = np.random.randn(100,4).astype(np.float32)
    act = np.random.randn(100,2).astype(np.float32)
    logp = np.random.randn(100).astype(np.float32)
    ret = np.random.randn(100).astype(np.float32)
    adv = np.random.randn(100).astype(np.float32)
    actor_data = dict(
        img = tf.convert_to_tensor(img, dtype=tf.float32),
        odom = tf.convert_to_tensor(odom, dtype=tf.float32),
        act = tf.convert_to_tensor(act, dtype=tf.float32),
        logp = tf.convert_to_tensor(logp, dtype=tf.float32),
        adv = tf.convert_to_tensor(adv, dtype=tf.float32)
    )
    critic_data = dict(
        img = tf.convert_to_tensor(img, dtype=tf.float32),
        odom = tf.convert_to_tensor(odom, dtype=tf.float32),
        ret = tf.convert_to_tensor(ret, dtype=tf.float32)
    )
    actor_dataset = tf.data.Dataset.from_tensor_slices(actor_data)
    critic_dataset = tf.data.Dataset.from_tensor_slices(critic_data)
    agent.train(actor_dataset, critic_dataset, 5)
