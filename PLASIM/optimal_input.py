import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm


class Regularizer():
    def __init__(self, l1coef=0, l2coef=0, rough_coef=0.1, gradient_regularizer=None, target_roughness=28, area_weights=1, track_components=True):
        self.l1coef = l1coef
        self.l2coef = l2coef

        self.gradient_regularizer = gradient_regularizer
        self.rough_coef = rough_coef if gradient_regularizer is not None else 0
        self.target_roughness = target_roughness

        self.area_weights = tf.convert_to_tensor(area_weights, dtype=tf.float32)

        self.info = {} if track_components else None

    def mean_no_batch(self, x):
        return tf.reduce_mean(x, axis=np.arange(1,len(x.shape)))

    def __call__(self, x):
        o = 0
        if self.l1coef:
            u = self.mean_no_batch(tf.abs(x)*self.area_weights)
            if self.info is not None:
                self.info['l1'] = u.numpy()
            o = o + self.l1coef*u
        if self.l2coef:
            u = self.mean_no_batch(x**2*self.area_weights)
            if self.info is not None:
                self.info['l2'] = np.sqrt(u.numpy())
            o = o + self.l2coef*u
        if self.rough_coef:
            u = tf.stack([self.gradient_regularizer(x[i]) for i in range(x.shape[0])])
            # u = tf.sqrt(self.gradient_regularizer(x[0]))
            if self.info is not None:
                self.info['roughness'] = u.numpy()
            o = o + self.rough_coef*(u - self.target_roughness)**2

        return o


class OptimalInput():
    def __init__(self, model, regularization, maxiter=100, lr=0.01, init_optimizer_every=None, physical_mask=None, ori_coef=0, weights=None):
        self.model = model
        self.regularization = regularization
        self.maxiter = maxiter
        self.lr = lr
        self.init_optimizer_every = init_optimizer_every
        self.physical_mask = physical_mask if physical_mask is not None else 1
        self.ori_coef = ori_coef
        self.weights = weights if weights is not None else 1

        self.opt_history = None
        self.info = None

    def toggle_history_tracking(self, track=False, verbose=False):
        if track:
            if verbose:
                print('History tracking enabled')
            self.opt_history = []
            self.info = {}
            self.regularization.info = {}
        else:
            if verbose:
                print('History tracking disabled')
            self.opt_history = None
            self.info = None
            self.regularization.info = None

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    @property
    def seed(self):
        return self.seed_.numpy()

    @property
    def optimized_input(self):
        return self.input.numpy()

    def init(self, seed):
        self.seed_ = tf.convert_to_tensor(seed, dtype=tf.float32)
        self.input = tf.Variable(self.seed_)
        self.init_optimizer()
        self.toggle_history_tracking(False)

    def loss_function(self, x=None):
        if x is not None:
            inim = tf.convert_to_tensor(x*self.physical_mask, dtype=tf.float32)
        else:
            inim = self.input*self.physical_mask

        o = self.model(inim)
        reg = self.regularization(inim)
        if self.ori_coef:
            u = tf.reduce_mean((inim - self.seed_)**2*self.area_weights)
            if self.info is not None:
                self.info['distance_from_seed'] = u.numpy()
            reg = reg + self.ori_coef*u

        loss = o + reg
        if self.info is not None:
            self.info['input'] = inim.numpy()
            self.info['output'] = o.numpy()
            self.info['loss'] = loss.numpy()
            self.info['regularization'] = reg.numpy()
            self.info.update(self.regularization.info)
            self.opt_history.append(self.info.copy())
        return loss


    def __call__(self, seed, track_history=False):
        self.init(seed)
        self.toggle_history_tracking(track_history, verbose=True)

        for i in tqdm(range(self.maxiter + 1)):
            if self.init_optimizer_every is not None and i % self.init_optimizer_every == 0 and i != 0:
                self.init_optimizer()


            # Perform one optimization step
            with tf.GradientTape() as tape:
                loss = self.loss_function()

            # compute gradients
            gradients = tape.gradient(loss, self.input)

            # decide whether to stop if the gradient is too small

            # update input image
            if i < self.maxiter:
                self.optimizer.apply_gradients([(gradients, self.input)])

        return self.optimized_input