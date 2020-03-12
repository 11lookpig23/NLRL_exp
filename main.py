from core.setup import *
from collections import OrderedDict
import argparse
import json

def generalized_test(task, name, algo,logstep,rep):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if task == "cliffwalking":
        env = CliffWalking
    elif task == "unstack":
        env = Unstack
    elif task == "stack":
        env = Stack
    elif task == "on":
        env = On
    elif task == "windycliffwalking":
        env = WindyCliffWalking
    elif task == "tic":
        print(" ==tic== ")
        env = TicTacTeo
    else:
        raise ValueError()
    import tensorflow as tf
    summary = OrderedDict()
    if algo=="DILP":
        starter = start_DILP
    else:
        pass
    variations = env.all_variations
    print("----++----",list(variations))
    for variation in [""]+list(variations):
        tf.reset_default_graph()
        print("==========="+variation+"==============")
        result = starter(task, name, "evaluate", logstep,rep,variation)
        pprint(result)
        variation = "train" if not variation else variation
        summary[variation] = {"mean":round(result["mean"], 3), "std": round(result["std"], 3),
                              "distribution":result["distribution"]}
    for k,v in summary.items():
        print(k+": "+str(v["mean"])+"+-"+str(v["std"]))


#@ray.remote
def start_DILP(task, name, mode, logstep, rep, variation=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if task == "unstack":
        man, env = setup_unstack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=50, name=name,log_steps=logstep,rep = rep)

    elif task == "tic":
        man, env = setup_tictacteo(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                    state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                    batched=True, steps=12000, name=name,log_steps=logstep,rep = rep)

    elif task == "stack":
        man, env = setup_stack(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=22002, name=name)
    elif task == "on":
        man, env = setup_on(variation)
        agent = RLDILP(man, env, state_encoding="atoms")
        if variation:
            critic = None
        else:
            critic = NeuralCritic([20], env.state_dim, 1.0, learning_rate=0.001,
                                  state2vector=env.state2vector, involve_steps=True)
        learner = ReinforceLearner(agent, env, 0.05, critic=critic,
                                   batched=True, steps=30000, name=name)
    else:
        raise ValueError()
    if mode == "train":
        return learner.train()#[-1]
    elif mode == "evaluate":
        return learner.evaluate()
    else:
        raise ValueError()

from pprint import pprint
if __name__ == "__main__":
    logstep = 800
    rep = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--task')
    parser.add_argument('--algo')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if args.mode=="generalize":
        generalized_test(args.task, args.name, args.algo,logstep,rep)
    elif args.mode=="train":
        try:
            if args.algo == "DILP":
                starter = start_DILP
            else:
                raise ValueError()
            pprint(starter(args.task, args.name, args.mode,logstep,rep))
        except Exception as e:
            #print(e.message)
            raise e
        finally:
         #   pprint(starter(args.task, args.name, "evaluate"))
            generalized_test(args.task, args.name, args.algo,logstep,rep)

