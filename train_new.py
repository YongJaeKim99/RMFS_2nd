from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameActNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from copy import deepcopy
import os
import random
import time
import sys
from model.PPO import PPO_initialize
from model.PPO import Memory

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

device = torch.device(configs.device)


class Trainer:
    def __init__(self, config):

        self.n_p = config.n_p
        self.n_t = config.n_t
        self.low = config.low
        self.high = config.high
        self.act_per_proj_min = int(0.8 * self.n_t)
        self.act_per_proj_max = int(1.2 * self.n_t)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        torch.set_default_dtype(torch.float32)
        if device.type == 'cuda':
            torch.set_default_device('cuda')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_p}x{self.n_t}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_p}x{self.n_t}{strToSuffix(config.data_suffix)}'

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameActNums(self.n_p, self.n_t)
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == 'SD1':
            self.vali_env = FJSPEnvForVariousOpNums(self.n_p, self.n_t)
        elif self.data_source == 'SD2':
            self.vali_env = FJSPEnvForSameActNums(self.n_p, self.n_t)

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_proj_length, dataset_act_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_proj_length, dataset_act_pt)
            else:
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_act=state.fea_act_tensor,  # [sz_b, N, 8]
                                                             act_mask=state.act_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, P]
                                                             fea_team=state.fea_team_tensor,  # [sz_b, T, 6]
                                                             team_mask=state.team_mask_tensor,  # [sz_b, T, T]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, T, T, P]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, P, T]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, P, T]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_act_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_act_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            ep_et = time.time()
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config, 
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl" 
        :return: new training instances
        """
        prepare_ProjLength = [random.randint(self.act_per_proj_min, self.act_per_proj_max) for _ in range(self.n_p)]
        dataset_ProjLength = []
        dataset_ActPT = []
        for i in range(self.num_envs):
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_p, self.n_t, self.act_per_proj_min, self.act_per_proj_max,
                                     nums_ope=prepare_ProjLength, path='./test', flag_doc=False)
                ProjLength, ActPT, _ = case.get_case(i)

            else:
                ProjLength, ActPT, _ = SD2_instance_generator(config=self.config)
            dataset_ProjLength.append(ProjLength)
            dataset_ActPT.append(ActPT)

        return dataset_ProjLength, dataset_ActPT

    def validate_envs_with_same_act_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of activities
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_act=state.fea_act_tensor,  # [sz_b, N, 8]
                                        act_mask=state.act_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, P]
                                        fea_team=state.fea_team_tensor,  # [sz_b, T, 6]
                                        team_mask=state.team_mask_tensor,  # [sz_b, T, T]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, T, T, P]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, P, T]
                                        fea_pairs=state.fea_pairs_tensor)  # [sz_b, P, T]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_act_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of activities
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_act=state.fea_act_tensor[batch_idx],  # [sz_b, N, 8]
                                        act_mask=state.act_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, P]
                                        fea_team=state.fea_team_tensor[batch_idx],  # [sz_b, T, 6]
                                        team_mask=state.team_mask_tensor[batch_idx],  # [sz_b, T, T]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, T, T, P]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, P, T]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, P, T]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False))


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
