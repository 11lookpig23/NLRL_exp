from core.induction import *
from copy import deepcopy
from collections import namedtuple
import pickle
import json
import numpy as np
import copy

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

Episode = namedtuple("Episode", ["reward_history", "action_history", "action_trajectory_prob", "state_history",
               "valuation_history", "valuation_index_history", "input_vector_history",
                                 "returns", "steps", "advantages", "final_return"])

class ReinforceLearner(object):
    def __init__(self, agent, enviornment, learning_rate, log_steps,rep,critic=None,
                 steps=300, name=None, discounting=1.0, batched=True, optimizer="RMSProp", end_by_episode=True,
                 minibatch_size=10):
        # super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        if isinstance(agent, RLDILP):
            self.type = "DILP"
        else:
            self.type = "Random"
        self.env = enviornment
        self.agent = agent
        self.state_encoding = agent.state_encoding
        self.learning_rate = learning_rate
        self._construct_train(learning_rate)
        self.critic=critic
        self.total_steps = steps
        self.name = name
        self.discounting = discounting
        self.batched = batched
        self.end_by_episode=end_by_episode
        self.batch_size = minibatch_size
        self.optimizer = optimizer
        self.log_steps = log_steps
        self.repeat = rep

    def _construct_train(self, learning_rate):
        self.tf_returns = tf.placeholder(shape=[None], dtype=tf.float32)
        #self.tf_episode_n = tf.placeholder(shape=[])
        self.tf_advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_additional_discount = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_actions_valuation_indexes = tf.placeholder(shape=[None, self.env.action_n], dtype=tf.int32)
        self.tf_action_index = tf.placeholder(shape=[None], dtype=tf.int32)
        self._construct_action_prob()
        indexed_action_prob = tf.batch_gather(self.tf_action_prob, self.tf_action_index[:, None])[:, 0]
        self.tf_loss = self.loss(indexed_action_prob)
        #self.tf_loss = tf.Print(self.tf_loss, [self.tf_loss])
        self.tf_gradients = tf.gradients(self.tf_loss, self.agent.all_variables())
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        try:
            self.tf_train = self.optimizer.minimize(self.tf_loss, tf.train.get_or_create_global_step(),
                                                    var_list=self.agent.all_variables())
        except Exception as e:
            # For random agent
            pass
        self.saver = tf.train.Saver()

    def loss(self, indexed_action_prob):
        rl_loss = (-tf.reduce_sum(tf.log(tf.clip_by_value(indexed_action_prob, 1e-5, 1.0))
               )*self.tf_advantage*self.tf_additional_discount)
        #excess_penalty = 0.01*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(self.tf_action_eval, axis=1)-1.0)**2)
        #regularization_loss = 1e-4*tf.reduce_mean(tf.stack([tf.nn.l2_loss(v) for v in self.agent.all_variables()]))
        #entropy_loss = tf.reduce_sum(self.tf_action_prob*tf.log(self.tf_action_prob))
        return rl_loss#+regularization_loss

    def _construct_action_prob(self):
        """
        this method implements the function $p_a$ in the paper
        """
        if self.type == "DILP":
            # slice the action valuations from the valuation vectors
            action_eval = tf.batch_gather(self.agent.tf_result_valuation, self.tf_actions_valuation_indexes)
            self.tf_action_eval = action_eval
            sum_action_eval = tf.tile(tf.reduce_sum(action_eval, axis=1, keepdims=True), [1, self.env.action_n])
            action_prob = tf.where(sum_action_eval > 1.0,
                                   action_eval / sum_action_eval,
                                   action_eval + (1.0 - sum_action_eval) / float(self.env.action_n))
            # action_prob = action_eval / sum_action_eval
            self.tf_action_prob = action_prob
        if self.type == "NN" or self.type=="Random":
            self.tf_action_prob = self.agent.tf_output

    def grad(self):
        loss_value = self.tf_loss
        weight_decay = 0.0
        regularization = 0
        for weights in self.agent.all_variables():
            weights = tf.nn.softmax(weights)
            regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
        loss_value += regularization/len(self.agent.all_variables())
        return tf.gradients(loss_value, self.agent.all_variables())

    def sample_episode(self, sess, max_steps=99999):
        action_prob_history = []
        action_history = []
        reward_history = []
        action_trajectory_prob = []
        valuation_history = []
        state_history = []
        input_vector_history = []
        valuation_index_history = []
        steps = []
        step = 0
        while step<max_steps:
            if self.type == "DILP":
                indexes = self.agent.get_valuation_indexes(self.env.state2atoms(self.env.state))
                inputs = None # inputs are needed only for neural network models, so this is none
                if self.state_encoding=="terms":
                    valuation = self.agent.base_valuation
                else:
                    valuation = self.agent.base_valuation + self.agent.axioms2valuation(self.env.state2atoms(self.env.state))
                action_prob,result = sess.run([self.tf_action_prob, self.agent.tf_result_valuation], feed_dict={self.agent.tf_input_valuation: [valuation],self.tf_actions_valuation_indexes: [indexes]})           
            
            action_prob = action_prob[0]
            actprob = copy.deepcopy(action_prob)
            #normact = actprob/np.sum(actprob)
            #print("normact---",normact)
            action_index = np.random.choice(range(self.env.action_n), p=action_prob)

            if self.state_encoding == "terms":
                action = self.env.action_index2atom(action_index)
            elif self.state_encoding == "atoms":
                action = self.agent.all_actions[action_index]
            else:
                if action_index<len(self.env.all_actions):
                    action = self.env.all_actions[action_index]
                else:
                    action = np.random.choice(self.env.all_actions)
            steps.append(step)
            state_history.append(self.env.state)
            ###
            # if env == tictactoe
            ###
            if self.env.__class__.__name__=='TicTacTeo':
                print('TicTacEnv----')
                reward, finished = self.env.next_step(actprob)
            else:
                reward, finished = self.env.next_step(action)
            reward_history.append(reward)
            action_history.append(action_index)
            action_trajectory_prob.append(action_prob[action_index])
            valuation_history.append(valuation)
            valuation_index_history.append(indexes)
            input_vector_history.append(inputs)
            step += 1
            if finished:
                self.env.reset()
                break
        final_return = [np.sum(reward_history)]
        returns = discount(reward_history, self.discounting)
        if self.critic:
            self.critic.batch_learn(state_history, reward_history, sess)
            values = self.critic.get_values(state_history,sess,steps).flatten()
            advantages = generalized_adv(reward_history, values, self.discounting)
            # advantages = np.array(returns) - values
        else:
            advantages = returns
        advantages[-1] = 0.0
        return Episode(reward_history, action_history, action_trajectory_prob, state_history,
               valuation_history, valuation_index_history, input_vector_history,
                       returns, steps, advantages, final_return)

    def get_minibatch_buffer(self, sess, batch_size=50, end_by_episode=True):
        empty_buffer = [[] for _ in range(10)]
        episode_buffer = deepcopy(empty_buffer)
        sample_related_indexes = range(10)

        def dump_episode2buffer(episode):
            for i in sample_related_indexes:
                episode_buffer[i].extend(episode[i])

        def split_buffer(raw_buffer, index):
            if len(episode_buffer[0]) < index:
                return raw_buffer, deepcopy(empty_buffer)
            result = []
            new_buffer = []
            for l in raw_buffer:
                result.append(l[:index])
                new_buffer.append(l[index:])
            return result, new_buffer

        while True:
            if len(episode_buffer[0]) ==0:
                if end_by_episode:
                    e = self.sample_episode(sess)
                    dump_episode2buffer(e)
                    final_return = e.final_return
                else:
                    while len(episode_buffer[0]) < batch_size:
                        e = self.sample_episode(sess)
                        dump_episode2buffer(e)
                        final_return = e.final_return
            result, episode_buffer = split_buffer(episode_buffer, batch_size)
            yield Episode(*(result+[final_return]))


    def summary_scalar(self, name, scalar):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name, scalar)

    def summary_histogram(self, name, data):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.histogram(name, data)

    def setup_train(self, sess, auto_load=True):
        sess.run([tf.initializers.global_variables()])
        if self.name:
            if auto_load:
                try:
                    path = "./model/" + self.name
                    self.load(sess, path)
                except Exception as e:
                    print(e)
            self.summary_writer = tf.contrib.summary.create_file_writer("./model/"+self.name, flush_millis=10000)
            self.summary_scalar("returns", self.tf_returns[0])
            self.summary_histogram("advantages", self.tf_advantage)
            self.summary_scalar("loss", self.tf_loss)
            #self.summary_histogram("weights", tf.concat(self.agent.all_variables(), axis=0))
            with self.summary_writer.as_default():
                tf.contrib.summary.initialize(graph=tf.get_default_graph(), session=sess)
        else:
            self.summary_writer = None
            # model definition code goes here
            # and in it call

    def evaluate(self, repeat=5):
        def printval(file,data):
            doc = open(file,'w')
            print(data,file = doc)
            doc.close()
        repeat = self.repeat
        results = []

        with tf.Session() as sess:
            Reli = []
            Actli = []
            Acttrajli = []
            Stateli = []
            self.setup_train(sess)
            self.agent.log(sess)
            rules = self.agent.get_predicates_definition(sess, threshold=0.05) if self.type == "DILP" else []
            for _ in range(repeat):
                e = self.sample_episode(sess)
                reward_history, action_history, action_prob_history, state_history, \
                valuation_history, valuation_index_history, input_vector_history, returns, steps, adv, final_return = e
                results.append(final_return)
                Reli.append(reward_history)
                Actli.append(action_history)
                Acttrajli.append(action_prob_history)
                Stateli.append(state_history)
        unique, counts = np.unique(results, return_counts=True)
        try:
            printval("reli,txt",np.array(Reli))
        except:
            print("Reli++++",Reli)
        try:
            printval("Actli.txt",np.array(Actli))
        except:
            print("Actli+++++++",Actli)
        try:
            printval("acttraj.txt",np.array(Acttrajli))
        except:
            print("Acttrajli+++++++++",Acttrajli)
        try:
            printval("state.txt",np.array(Stateli))
        except:
            print("Stateli+++++++++++++",Stateli)
        distribution =  dict(zip(unique, counts))
        return {"distribution": distribution, "mean": np.mean(results), "std": np.std(results),
                "min": np.min(results), "max": np.max(results), "rules": rules}

    def train_step(self, sess):
        e = next(self.minibatch_buffer)
        #e = self.sample_episode(sess)
        reward_history, action_history, action_prob_history, state_history,\
            valuation_history, valuation_index_history, input_vector_history,\
            returns, steps, advantage, final_return = e
        #additional_discount = np.cumprod(self.discounting*np.ones_like(advnatage))
        #advantage = normalize(advantage)
        additional_discount = np.ones_like(advantage)
        log = {"return":final_return[0], "action_history":[str(self.env.all_actions[action_index])
                                                               for action_index in action_history]}

        if self.batched:
            ops = [self.tf_train, tf.contrib.summary.all_summary_ops(), self.tf_gradients] if self.name else [self.tf_train]
            if self.type == "DILP":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                                 self.tf_returns:final_return,
                                 self.tf_action_index:np.array(action_history),
                                 self.tf_actions_valuation_indexes: np.array(valuation_index_history),
                                 self.agent.tf_input_valuation: np.array(valuation_history)}
        else:
            first = True
            for action_index, adv, acc_discount, val, val_index in zip(action_history, advantage, additional_discount,
                                                          valuation_history,valuation_index_history):
                ops = [self.tf_train, self.tf_loss, self.tf_action_prob]
                if first == True and self.name:
                    ops += [tf.contrib.summary.all_summary_ops()]
                    first = False
                result = sess.run(ops, {self.tf_advantage: [adv],
                                 self.tf_additional_discount: [acc_discount],
                               self.tf_returns: final_return,
                               self.tf_action_index: [action_index],
                               self.tf_actions_valuation_indexes: [val_index],
                               self.agent.tf_input_valuation: [val]})
        return log

    def save(self, sess, path):
        self.saver.save(sess, path + "/parameters.ckpt")

    def load(self, sess, path):
        self.saver.restore(sess, path+"/parameters.ckpt")

    def train(self):
        with tf.Session() as sess:
            trainRes = []
            self.setup_train(sess)
            self.minibatch_buffer = self.get_minibatch_buffer(sess, batch_size=self.batch_size,
                                                              end_by_episode=self.end_by_episode)
            for i in range(self.total_steps):
                log = self.train_step(sess)
                trainRes.append(log["return"])

                print("-"*20)
                print("step "+str(i)+"return is "+str(log["return"]))
                if (i+1)%self.log_steps==0:
                    self.agent.log(sess)
                    if self.name:
                        path = "./model/" + self.name
                        self.save(sess, path)
                    pprint(log)
                print("-"*20+"\n")
            np.savetxt("trainres.txt",np.array(trainRes))
        return log["return"]

class NeuralCritic(object):
    def __init__(self, unit_list, state_size, discounting, learning_rate, state2vector,
                 involve_steps=False):
        self.unit_list = unit_list
        self.state2vector = state2vector
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.tf_steps = tf.placeholder(dtype=tf.float32, shape=[None])
        if involve_steps:
            outputs = tf.concat([self.tf_input, self.tf_steps[:, np.newaxis]], axis=1)
        else:
            outputs = self.tf_input
        with tf.variable_scope("critic"):
            for unit_n in unit_list:
                outputs = tf.layers.dense(outputs, unit_n, activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, 1)
        self.involve_steps = involve_steps
        self.tf_output = outputs
        self.state_encoding = "vector"
        self.discounting = discounting
        self.tf_reward = tf.placeholder(dtype=tf.float32, shape=[None])
        self.tf_returns = tf.placeholder(dtype=tf.float32, shape=[None])
        self.tf_loss = tf.reduce_sum(tf.square(self.tf_output[:, 0] - self.tf_returns))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.tf_train = self.optimizer.minimize(self.tf_loss, tf.train.get_or_create_global_step(),
                                                var_list=self.all_variables())


    def all_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic")

    def log(self, sess):
        pass

    def batch_learn(self, states, rewards, sess):
        states = [self.state2vector(s) for s in states]
        returns = discount(rewards, self.discounting)
        sess.run([self.tf_train], feed_dict={self.tf_input:states, self.tf_reward: rewards,
                                             self.tf_returns: returns,
                                             self.tf_steps:np.array(range(0, len(states)),
                                                                    dtype=np.float32)})

    def get_values(self, states, sess, steps=None):
        states = [self.state2vector(s) for s in states]
        if self.involve_steps:
            return sess.run([self.tf_output], feed_dict={self.tf_input:np.array(states),
                                                     self.tf_steps: np.array(steps, dtype=np.float32)})[0]
        else:
            return sess.run([self.tf_output], feed_dict={self.tf_input:np.array(states)})[0]

def discount(r, discounting):
    discounted_reward = np.zeros_like(r, dtype=np.float32)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * discounting + r[i]
        discounted_reward[i] = G
    return discounted_reward

def normalize(scalars):
    mean, std = np.mean(scalars), np.std(scalars)
    return (scalars - mean)/std

def generalized_adv(rewards, values, discounting, lam=0.95):
    values[-1] = rewards[-1]
    deltas = rewards[:-1] + discounting * values[1:] - values[:-1]
    return np.concatenate([discount(deltas, discounting*lam), [0]], axis=0)