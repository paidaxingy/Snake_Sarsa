import numpy as np
import random
import pygame
import pickle
from collections import deque
import matplotlib.pyplot as plt
# 初始化 Pygame
pygame.init()

# 游戏设置
GRID_SIZE = 20  # 每个格子的像素大小
GRID_COUNT = 10  # 网格数量
SCREEN_SIZE = GRID_SIZE * GRID_COUNT
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Q 表保存文件名
Q_TABLE_FILE = "main.pkl"

# 贪吃蛇环境
class SnakeEnv:
    def __init__(self, grid_count=GRID_COUNT):
        self.grid_count = grid_count
        self.reset()

    def reset(self):
        self.snake = deque([[GRID_COUNT//2, GRID_COUNT//2]])  # 初始蛇头位置
        self.food = self._generate_food()
        self.food_num = 0
        self.done = False
        self.current_direction = 3  # 初始方向：右（0=上, 1=下, 2=左, 3=右）
        return self._get_state()

    def _generate_food(self):
        corners = [
            (0, 0),  # 左上角
            (0, self.grid_count - 1),  # 右上角
            (self.grid_count - 1, 0),  # 左下角
            (self.grid_count - 1, self.grid_count - 1)  # 右下角
        ]
        while True:
            food = [random.randint(0, self.grid_count - 1), random.randint(0, self.grid_count - 1)]
            # 检查生成的食物不在蛇身上，也不在角落
            if food not in self.snake and tuple(food) not in corners:
                return food


    def _get_state(self):
        head = self.snake[0]
        food_direction = [
            self.food[0] - head[0],  # 食物在蛇头上/下的相对距离
            self.food[1] - head[1]   # 食物在蛇头左/右的相对距离
        ]
        local_view = self._get_local_view(head)
        return (tuple(food_direction), tuple(local_view.flatten()))

    def _get_local_view(self, head):
        """提取蛇头附近 3x3 区域状态"""
        local_view = np.zeros((3, 3), dtype=int)
        for i in range(-1, 2):
            for j in range(-1, 2):
                x, y = head[0] + i, head[1] + j
                if 0 <= x < self.grid_count and 0 <= y < self.grid_count and [x, y] in self.snake:
                    local_view[i + 1, j + 1] = 1
        return local_view

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done

        # 防止 180 度回头
        if (self.current_direction == 0 and action == 1) or \
           (self.current_direction == 1 and action == 0) or \
           (self.current_direction == 2 and action == 3) or \
           (self.current_direction == 3 and action == 2):
            action = self.current_direction  # 忽略非法动作，保持当前方向

        # 更新当前方向
        self.current_direction = action

        # 动作定义：0=上, 1=下, 2=左, 3=右
        head = self.snake[0]
        new_head = [head[0], head[1]]
        if action == 0:  # 上
            new_head[0] -= 1
        elif action == 1:  # 下
            new_head[0] += 1
        elif action == 2:  # 左
            new_head[1] -= 1
        elif action == 3:  # 右
            new_head[1] += 1

        # 检查是否撞墙或撞到自己
        if (
            new_head[0] < 0 or new_head[0] >= self.grid_count or
            new_head[1] < 0 or new_head[1] >= self.grid_count or
            new_head in self.snake
        ):
            self.done = True
            return self._get_state(), -100, self.done

        # 更新蛇的位置
        self.snake.appendleft(new_head)
        reward = -2  # 默认移动惩罚

        # 检查是否吃到食物
        if new_head == self.food:
            reward = 200  # 吃食物奖励
            self.food = self._generate_food()
            self.food_num += 1
        else:
            self.snake.pop()  # 没有吃到食物时移除蛇尾

        return self._get_state(), reward, self.done

    def render(self, screen):
        screen.fill(BLACK)

        # 绘制食物
        pygame.draw.rect(screen, RED, (self.food[1] * GRID_SIZE, self.food[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # 绘制蛇
        for segment in self.snake:
            pygame.draw.rect(screen, GREEN, (segment[1] * GRID_SIZE, segment[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        pygame.display.flip()


class SarsaAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, epsilon_decay=0.99995, epsilon_min=0.001, n_steps=4, heuristic_weight = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_steps = n_steps
        self.q_table = self._load_q_table()
        self.heuristic_weight = heuristic_weight

    def _load_q_table(self):
        try:
            with open(Q_TABLE_FILE, "rb") as f:
                print("加载 Q 表成功！")
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            print("未找到 Q 表文件，初始化新的 Q 表...")
            return {}

    def save_q_table(self):
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(self.q_table, f)
            print("Q 表已保存到文件。")

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, env):
        """
        筛选可行动作并优先选择最优动作。
        """
        legal_actions = self._get_legal_actions(env)
        if not legal_actions:
            return env.current_direction

        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)

        q_values = {a: self.get_q_value(state, a) for a in legal_actions}
        return max(q_values, key=q_values.get)

    def _get_legal_actions(self, env):
        """
        计算合法动作列表，同时预测三步路径安全性。
        """
        legal_actions = []
        head = env.snake[0]

        for action in range(self.action_size):
            new_head = [head[0], head[1]]
            if action == 0:  # 上
                new_head[0] -= 1
            elif action == 1:  # 下
                new_head[0] += 1
            elif action == 2:  # 左
                new_head[1] -= 1
            elif action == 3:  # 右
                new_head[1] += 1

            if self._is_valid(env, new_head) and self._predict_safe(env, new_head, self.n_steps):
                legal_actions.append(action)

        return legal_actions

    def _is_valid(self, env, position):
        """
        检查给定位置是否在合法范围内且不与蛇身体重叠。
        """
        return (
            0 <= position[0] < env.grid_count and
            0 <= position[1] < env.grid_count and
            position not in env.snake
        )

    def _predict_safe(self, env, head, steps):
        """
        递归预测接下来 steps 步是否安全。
        """
        if steps == 0:
            return True

        # 复制蛇的状态
        snake_copy = deque(env.snake)
        snake_copy.appendleft(head)
        snake_copy.pop()  # 模拟移动，移除蛇尾

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_head = [head[0] + dx, head[1] + dy]
            if self._is_valid(env, new_head):
                if self._predict_safe(env, new_head, steps - 1):
                    return True

        return False


    def update_q_value(self, state, action, reward, next_state, next_action, heuristic=0):
        """
        使用启发式引导的 Q 值更新，同时加入额外奖励/惩罚机制。
        """
        current_distance = abs(state[0][0]) + abs(state[0][1])
        next_distance = abs(next_state[0][0]) + abs(next_state[0][1])

        if next_distance < current_distance:
            reward += 1
        elif next_distance >= current_distance:
            reward -= 200

        heuristic = -next_distance
        target = reward + self.discount_factor * (self.get_q_value(next_state, next_action) + heuristic)
        current_q = self.get_q_value(state, action)
        # 更新 Q 值
        self.q_table[(state, action)] = current_q + self.learning_rate * (target - current_q)
    def update_n_step_q_values(self, state, action, reward, next_state, next_action, trajectory, done):
        """
        n-step SARSA Q 值更新，加入曼哈顿距离启发式
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前即时奖励
        :param next_state: 下一状态
        :param next_action: 下一动作
        :param trajectory: 经验存储队列（存储最近 n 步的 (s, a, r)）
        :param done: 是否为终止状态
        """
        # 将当前经验存入队列
        trajectory.append((state, action, reward))

        # 当轨迹长度达到 n 或游戏结束时，开始更新 Q 值
        if len(trajectory) >= self.n_steps or done:
            # 计算累计折扣奖励 G
            G = 0
            for i, (s, _, r) in enumerate(trajectory):
                # 引入启发式函数 H(s)，权重为 lambda
                food_direction = s[0]  # 食物相对蛇头的方向
                manhattan_distance = abs(food_direction[0]) + abs(food_direction[1])  # Δx + Δy
                heuristic_penalty = self.heuristic_weight * manhattan_distance
                G += (self.discount_factor ** i) * (r - heuristic_penalty)

            # 如果未结束，累加 n 步后的 Q 值预测
            if not done and len(trajectory) == self.n_steps:
                s_n, a_n, _ = trajectory[-1]
                G += (self.discount_factor ** self.n_steps) * self.get_q_value(s_n, a_n)

            # 更新轨迹中的第一个状态和动作的 Q 值
            s_0, a_0, _ = trajectory[0]
            current_q = self.get_q_value(s_0, a_0)
            self.q_table[(s_0, a_0)] = current_q + self.learning_rate * (G - current_q)

            # 移除队列中的第一个元素，保持队列长度为 n
            if len(trajectory) == self.n_steps:
                trajectory.pop(0)



    def decay_epsilon(self, env):
        """
        动态调整 epsilon，根据蛇的长度减少探索率。
        """
        snake_length = len(env.snake)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 动态调整，蛇越长减少探索
        if snake_length > 10:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9)


# 训练主函数
inf = 100000000000
def train_nsteps_snake():
    episodes = 1500  # 训练回合数
    max_food_num = 0
    food_num_history = []  # 每回合食物数量记录
    env = SnakeEnv()  # 初始化环境
    agent = SarsaAgent(state_size=11, action_size=4, n_steps=4, heuristic_weight=1)  # 使用 n-step SARSA，设置 n=4
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))  # 渲染窗口
    clock = pygame.time.Clock()
    render_interval = 300  # 每 render_interval 回合渲染一次
    trajectory = []  # 用于存储最近 n 步的 (s, a, r) 经验
    avg_num = 0.0
    for e in range(episodes):
        state = env.reset()
        action = agent.choose_action(state, env)  # 初始动作选择
        total_reward = 0

        while True:
            # 处理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # 保存 Q 表并退出
                    # agent.save_q_table()
                    pygame.quit()
                    print(f"max_food_num: {max_food_num}")
                    plot_food_num(food_num_history)
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # 保存 Q 表并退出
                    # agent.save_q_table()
                    pygame.quit()
                    print(f"max_food_num: {max_food_num}")
                    plot_food_num(food_num_history)
                    exit()

            # 环境交互
            next_state, reward, done = env.step(action)  # 执行动作
            next_action = agent.choose_action(next_state, env)  # 下一步动作选择

            # n-step Q 值更新
            agent.update_n_step_q_values(
                state, action, reward, next_state, next_action, trajectory, done
            )

            # 更新状态与动作
            state, action = next_state, next_action
            total_reward += reward
            
            # 渲染与帧率控制
            if e % render_interval == 0:
                env.render(screen)
                clock.tick(80)  # 帧率限制
            else:
                clock.tick(1000000000000)  # 无渲染模式，快速训练
                

            if done:
                # 游戏结束时，清空队列中剩余的 n 步并更新 Q 值
                while len(trajectory) > 0:
                    state, action, reward = trajectory.pop(0)  # 获取队首
                    # 计算剩余轨迹的折扣奖励
                    G = reward
                    for i, (s, _, r) in enumerate(trajectory):
                        food_direction = s[0]  # 食物相对蛇头的方向
                        manhattan_distance = abs(food_direction[0]) + abs(food_direction[1])  # Δx + Δy
                        heuristic_penalty = agent.heuristic_weight * manhattan_distance
                        G += (agent.discount_factor ** (i + 1)) * (r - heuristic_penalty)
                    # 最终更新 Q 值
                    current_q = agent.get_q_value(state, action)
                    agent.q_table[(state, action)] = current_q + agent.learning_rate * (G - current_q)

                agent.decay_epsilon(env)  # 衰减探索率
                break

        # 记录本回合食物数量
        max_food_num = max(max_food_num, env.food_num)
        avg_num += env.food_num
        food_num_history.append(env.food_num)

        # 定期打印进度并保存 Q 表
        if e % render_interval == 0:
            # agent.save_q_table()
            print(
                f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, "
                f"Epsilon: {agent.epsilon:.3f}, food_num: {env.food_num}, max_food_num: {max_food_num}"
            )

    # 训练完成后的统计与可视化
    print(f"max_food_num: {max_food_num}, Average: {avg_num / episodes:.3f}")
    plot_food_num(food_num_history)

def plot_food_num(food_num_history):
    plt.figure(figsize=(10, 6))
    plt.plot(food_num_history, label="Food Collected")
    plt.xlabel("Episodes")
    plt.ylabel("Food Number")
    plt.title("Food Number per Episode During Training")
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    train_nsteps_snake()