import os, sys, random
import numpy as np
from core import neuroevolution as utils_ne
from core import mod_utils as utils
import gym, torch
from core.hac import HAC
from core.hac_general import  HAC_general
from core.gym_wrapper import GymWrapper
from core.gym_fetch_wrapper import FetchWrapper


os.environ["CUDA_VISIBLE_DEVICES"]='3'
# os.environ["LD_LIBRARY_PATH"] = '$LD_LIBRARY_PATH:/nfs/site/home/shauhard/.mujoco/mjpro150/bin:/usr/lib/nvidia-384'
#os.environ["PATH"] = '~/anaconda3/bin:$PATH'
#os.environ["LD_LIBRARY_PATH"] ='$LD_LIBRARY_PATH:/usr/lib/nvidia-384'
# from subprocess import call
# call(['bash', '/nfs/site/home/shauhard/start.sh'])
#os.environ["PATH"] = '~/anaconda3/bin:$PATH'


#Pick Env
if True:
    ###################OpenAI Mujoco ####################
    #env_tag = 'Pendulum-v0'
    #env_tag = 'MountainCarContinuous-v0'
    #env_tag = 'Hopper-v2'
    #env_tag = 'Humanoid-v2'
    #env_tag = 'Ant-v2'
    # env_tag = 'HumanoidStandup-v2'
    #env_tag = 'Reacher-v2'
    #env_tag = 'Swimmer-v2'
    # env_tag = 'Walker2D-v2'
    # env_tag = 'InvertedPendulum-v2'
    #env_tag = 'HalfCheetah-v2'

    #################OpenAI Classic ######################
    #env_tag = 'BipedalWalker-v2'
    #env_tag = 'CarRacing-v0'

    #######################Open AI Robotics ################
    #env_tag = 'FetchReach-v1'
    #env_tag = 'FetchPush-v1'
    env_tag = 'FetchPickAndPlace-v1'


    #######################DM Suite ########################
    #env_name = "cheetah"; task_name = "run"
    #env_name = "humanoid"; task_name = "run"
    #env_name = "acrobot"; task_name = "swingup"

class Core:
    def __init__(self, args, env, tracker):
        self.env = env
        self.tracker = tracker


class Parameters:
    def __init__(self):

        #MetaParams
        self.num_games = 1000000
        self.use_rl = True
        self.use_evo = True
        self.is_cuda = True
        self.num_test_evals = 5

        #RL params
        self.use_ln = True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 4
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.is_mem_cuda = True

        #NeuroEvolution stuff
        self.pop_size = 10
        self.num_evals = 1
        self.elite_fraction = 0.1
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.extinction_prob = 0.005 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_ERL/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)

        #Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))

        #Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()

        #Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.ounoise = ddpg.OUNoise(args.action_dim)

        #Trackers
        self.num_games = 0; self.num_frames = 0; self.gen_frames = None



    def rollout(self, net, is_render, is_action_noise=False, store_transition=True):
        total_reward = 0.0

        state = self.env.reset()
        #if is_robotics_env: state = utils.odict_to_numpy(state)

        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda: state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            action = net.forward(state)
            action.clamp(-1,1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()

            next_state, reward, done, info = self.env.step(action.flatten())  #Simulate one step in environment
            #if is_robotics_env: next_state = utils.odict_to_numpy(next_state)
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition: self.add_experience(state, action, next_state, reward, done)
            state = next_state
        if store_transition: self.num_games += 1

        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        self.gen_frames = 0

        if self.args.use_evo: #No NEURO_EVO [RUN RL IN ISOLATION]
            ####################### N-EVO STUFF #####################
            all_fitness = []
            #Evaluate genomes/individuals
            for net in self.pop:
                fitness = 0.0
                for eval in range(self.args.num_evals): fitness += self.rollout(net, is_render=False, is_action_noise=False)
                all_fitness.append(fitness/self.args.num_evals)

            best_train_fitness = max(all_fitness)

            #Validation test
            champ_index = all_fitness.index(max(all_fitness))
            test_score = 0.0
            for eval in range(self.args.num_test_evals): test_score += self.rollout(self.pop[champ_index], is_render=True, is_action_noise=False, store_transition=False)/self.args.num_test_evals

            #NeuroEvolution's probabilistic selection and recombination step
            elite_index, worst_index = self.evolver.epoch(self.pop, all_fitness)

        else: test_score = None; best_train_fitness = None; elite_index = None


        ####################### RL-Stuff #########################
        if self.args.use_rl:
            #RL Experience Collection
            self.rollout(self.rl_agent.actor, is_render=False, is_action_noise=True) #Train

            #Test with no experience storing (Pure DDPG)
            rl_score = 0.0
            for eval in range(self.args.num_test_evals):
                rl_score += self.rollout(self.rl_agent.actor, is_render=True, is_action_noise=False, store_transition=False)/self.args.num_test_evals

            #RL learning step
            if len(self.replay_buffer) > self.args.batch_size * 5:
                for _ in range(int(self.gen_frames*self.args.frac_frames_train)):
                    batch = self.replay_buffer.sample(self.args.batch_size)
                    self.rl_agent.update_parameters(batch)

                #Synch RL Agent to NE
                if self.num_games % 10 == 0 and self.args.use_evo:
                    self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                    print('Synch from RL --> Nevo')

        else: rl_score = None

        return best_train_fitness, test_score, rl_score, elite_index

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = utils.Tracker(parameters, ['score', 'steps'], '_score.csv')  # Initiate tracker
    #frame_tracker = utils.Tracker(parameters, ['frame_evo', 'frame_rl'], '_score.csv')  # Initiate tracker
    #time_tracker = utils.Tracker(parameters, ['time_evo', 'time_rl'], '_score.csv')

    if False: #Deepmind Suite
        env = suite.load(domain_name=env_name, task_name=task_name)
        parameters.action_dim = env.action_spec().shape[0]
        state = env.observation_spec()
        shape = 0
        for key, value in state.items():
            if len(value.shape) != 0: shape+= value.shape[0]
            else: shape += 1
        parameters.state_dim = shape

    else: #OpenAI
        env = gym.make(env_tag)
        is_robotics = isinstance(env.observation_space.spaces, dict)
        if is_robotics: #Robotics
            parameters.goal_dim = int(env.observation_space.spaces['desired_goal'].shape[0])
            env = FetchWrapper(env, dict_keys=["observation","desired_goal"])

        else: #Normal Env
            env = GymWrapper(env) #Normal Wrapper
        parameters.action_dim = int(env.action_space.shape[0])
        parameters.state_dim = int(env.observation_space.shape[0])
        env.seed(parameters.seed)

    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    core = Core(parameters, env, tracker)
    print ('Running', env_tag, 'with S in',parameters.state_dim, 'dim, A in', parameters.action_dim, 'dim, and G in', parameters.goal_dim, 'dim')
    if is_robotics:  # Robotics
        agent = HAC(parameters, core)
    else:
        agent = HAC_general(parameters, core)
    agent.train(tracker)




    # agent = Agent(parameters, env)
    # print('Running', env_name+' '+task_name if is_dm_suite else env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using', 'ERL'if parameters.use_evo and parameters.use_rl else 'RL')
    #
    # next_save = 100; time_start = time.time()
    # while agent.num_games <= parameters.num_games:
    #     best_train_fitness, evo_score, rl_score, elite_index = agent.train()
    #     print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Ep_best:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Evo_Score:','%.2f'%evo_score if evo_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], ' RL_Score:', '%.2f' %rl_score if rl_score!=None else None, ' Avg:','%.2f'%tracker.all_tracker[1][1])
    #     tracker.update([evo_score, rl_score], agent.num_games)
    #     frame_tracker.update([evo_score, rl_score], agent.num_frames)
    #     time_tracker.update([evo_score, rl_score], time.time()-time_start)
    #
    #     #Save
    #     if agent.num_games > next_save:
    #         next_save += 100
    #         torch.save(agent.rl_agent.actor.state_dict(), parameters.save_foldername+'rl_net')
    #         if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net')
    #         print("Progress Saved")











