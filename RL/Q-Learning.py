import numpy as np
import pandas as pd
import time

np.random.seed(0)  # 设置随机数

N_STATES = 6  # 代表有6个状态，包括5个空地，一个宝藏地
STATES = ['EMPTY__1', 'EMPTY__2', 'EMPTY__3', 'EMPTY__4', 'EMPTY__5', 'TREASURE']
ACTIONS = ['LEFT', 'RIGHT']  # 有哪些行为
EPSILON = 0.9
ALPHA = 0.1  # 学习速率
GAMMA = 0.9  # 衰减率discount
MAX_EPISODES = 20  # 最多走20回合
FRESH_TIME = 0.3  # 0.3s走一步，方便演示

'''
初始化Q表
'''


def create_table(n_states, actions):
    # 0随机矩阵
    data = np.random.uniform(0.0, 0.001, (n_states, len(actions)))
    table = pd.DataFrame(data, index=STATES, columns=actions)
    return table


'''
根据状态和Q-Table来选择行为
'''


def choose_action(state, q_table):
    # 该状态下的某行为价值
    state_actions = q_table.loc[state, :]
    if np.random.uniform() > EPSILON:
        # 随机策略
        action_name = np.random.choice(ACTIONS)
    else:
        # 贪婪策略
        action_name = state_actions.idxmax()
    return action_name


'''
根据状态和行为获得下一个状态和即时奖励
如果下一个是终点状态，那么返回的状态是'TREASURE'
'''


def get_feedback(S, A):
    reward = -0.1  # 不建议多走路

    index_of_S = STATES.index(S)
    if A == 'RIGHT':
        # 是否终止
        if S == 'EMPTY__5':
            next_S = STATES[N_STATES - 1]
            reward += 1  # 奖励一下
        else:
            next_S = STATES[index_of_S + 1]
    else:
        if S == 'EMPTY__1':
            # 撞墙
            next_S = S
            # 撞墙是愚蠢的行为，惩罚
            reward = reward - 1
        else:
            next_S = STATES[index_of_S - 1]

    return next_S, reward


'''
打印迷宫场景
'''


def print_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'TREASURE':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[STATES.index(S)] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


'''
打印这一段经历，输出Q表
'''


def print_episode_conclusion(q_table, episode, steps):
    print('\r', end='')
    print(('-' * 10 + 'Episode {}' + '-' * 10).format(episode+1))
    print('Total_steps: {}'.format(steps))
    print('Q-Table:')
    print(q_table)
    print('\n')


'''
主流程
'''


def run_RL(n_episodes):
    # 创建Q表
    Q_Table = create_table(N_STATES, ACTIONS)
    print('-' * 10 + 'Initialize Q-Table' + '-' * 10)
    print(Q_Table)
    print()

    for episode in range(n_episodes):

        # 更改epsilon的值
        global EPSILON
        EPSILON = (episode+1)/5
        if EPSILON >= 1:
            EPSILON = 0.95


        step_count = 0
        S = STATES[0]
        is_terminated = False
        print_env(S, episode, step_count)
        while not is_terminated:

            A = choose_action(S, Q_Table)
            step_count += 1
            next_S, reward = get_feedback(S, A)

            # 更新Q值
            q_now = Q_Table.loc[S, A]
            if next_S != 'TREASURE':
                q_new = reward + GAMMA * Q_Table.loc[next_S, :].max()
            else:
                q_new = reward
                is_terminated = True

            Q_Table.loc[S, A] += ALPHA * (q_new - q_now)

            # 更新状态
            S = next_S

            # 打印迷宫
            print_env(S, episode, step_count)

        # 总结一下该回合
        print_episode_conclusion(Q_Table, episode, step_count)

    return Q_Table


if __name__ == '__main__':
    run_RL(15)
