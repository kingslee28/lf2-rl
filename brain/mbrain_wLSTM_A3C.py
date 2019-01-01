import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BATCH_SIZE, TARGET_REPLACE_ITER, MEMORY_CAPACITY, E_GREEDY

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# DQN
class Net(nn.Module):
    def __init__(self, action_n, state_n):
        super(Net, self).__init__()
        picture_n, feature_n = state_n[0], state_n[1]

        # input (4, 80, 190)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=3).to(device)  # (32, 20, 48)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1).to(device)  # (64, 10, 24)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1).to(device)  # (64, 10, 24)
        self.fc1 = nn.Linear(15360, 20).to(device)
        self.fc1.weight.data.normal_(0, 0.1)

        # LSTM
        hidden_n = 20
        self.lstm = nn.LSTMCell(20 + feature_n + action_n + 1, hidden_n, bias=False).to(device)
        self.hx = torch.randn(1, hidden_n)
        self.cx = torch.randn(1, hidden_n)

        # A3C
        self.actor_linear = nn.Linear(hidden_n, action_n).to(device)
        self.actor_linear.weight.data.normal_(0, 0.01)  # initialization
        self.ax = torch.randn(1, action_n)
        self.critic_linear = nn.Linear(hidden_n, 1).to(device)
        self.critic_linear.weight.data.normal_(0, 1.0)  # initialization
        self.crx = torch.randn(1, 1)

        if USE_CUDA:
            self.hx = self.hx.cuda()
            self.cx = self.cx.cuda()
            self.ax = self.ax.cuda()
            self.crx = self.crx.cuda()

        # output
        self.out = nn.Linear(hidden_n, action_n).to(device)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        picture, feature = x[0], x[1]
        # print("picture len: {}".format(len(picture)))
        # print("feature len: {}".format(len(feature)))

        # picture cnn
        picture = F.relu(self.conv1(picture))
        picture = F.relu(self.conv2(picture))
        picture = F.relu(self.conv3(picture))
        picture = picture.view(picture.size(0), -1)
        picture = F.relu(self.fc1(picture))
        # print("x shape {}".format(x.shape))

        # LSTM
        size = len(picture)
        # print("size: {}".format(size))
        if size == 1:
            x = torch.cat((picture, feature, self.ax, self.crx), 1)
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
            x = self.hx
            self.ax = self.actor_linear(x)
            self.crx = self.critic_linear(x)
        else:
            ax = torch.cat([self.ax]*BATCH_SIZE, 0)
            crx = torch.cat([self.crx]*BATCH_SIZE, 0)
            x = torch.cat((picture, feature, ax, crx), 1)
            x, _ = self.lstm(x, (torch.cat([self.hx]*BATCH_SIZE, 0), torch.cat([self.cx]*BATCH_SIZE, 0)))
        # print("x shape {}".format(x.shape))
        # print("=" * 10)

        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9):
        self.eval_net, self.target_net = Net(action_n=action_n, state_n=state_n), Net(action_n=action_n,
                                                                                      state_n=state_n)

        self.action_n = action_n
        self.state_n = state_n  # [(160, 380), 28]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = E_GREEDY
        self.env_shape = env_shape

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros(
            (MEMORY_CAPACITY, (4 * self.state_n[0][0] * self.state_n[0][1] + self.state_n[1]) * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.backward_count = 0

    def choose_action(self, x):
        picture = torch.unsqueeze(torch.FloatTensor(x[0]).to(device), 0)
        feature = torch.unsqueeze(torch.FloatTensor(x[1]).to(device), 0)
        x = [picture, feature]
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            # print("choose action and forward")
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)
        else:  # random
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.append(np.reshape(s[0], -1), s[1])  # [(4, 160, 380), 28] -> 4 * 160 * 380 + 28
        s_ = np.append(np.reshape(s_[0], -1), s_[1])
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("backward")
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        picture_idx = 4 * self.state_n[0][0] * self.state_n[0][1]
        feature_idx = self.state_n[1]
        state_idx = picture_idx + feature_idx  # 4 * 160 * 380 + 28
        b_picture = torch.FloatTensor(b_memory[:, :picture_idx]).to(device)
        b_feature = torch.FloatTensor(b_memory[:, picture_idx:state_idx]).to(device)
        b_a = torch.LongTensor(b_memory[:, state_idx:state_idx + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, state_idx + 1:state_idx + 2]).to(device)
        b_picture_ = torch.FloatTensor(b_memory[:, -state_idx:-feature_idx]).to(device)
        b_feature_ = torch.FloatTensor(b_memory[:, -feature_idx:]).to(device)
        # reshape (batch_size, 4, 160, 380)
        b_picture = np.reshape(b_picture, (BATCH_SIZE, 4, self.state_n[0][0], self.state_n[0][1])).to(device)
        b_picture_ = np.reshape(b_picture_, (BATCH_SIZE, 4, self.state_n[0][0], self.state_n[0][1])).to(device)
        b_s = [b_picture, b_feature]
        b_s_ = [b_picture_, b_feature_]

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        if self.backward_count > 0:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(),
                   'model/DQN/mix_eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))
        torch.save(self.target_net.state_dict(),
                   'model/DQN/mix_target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))

    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name))