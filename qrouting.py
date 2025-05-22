import numpy as np
import random

class QRouter:
    def __init__(self, num_routers, num_links, learning_rate, discount_factor):
        self.num_routers = num_routers
        self.num_links = num_links
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((num_routers, num_links))

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_links - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def get_q_value(self, state, action):
        return self.q_table[state, action]

class Network:
    def __init__(self, num_routers, num_links):
        self.num_routers = num_routers
        self.num_links = num_links
        self.delay = np.random.uniform(0, 10, size=(num_routers, num_links))
        self.throughput = np.random.uniform(0, 10, size=(num_routers, num_links))

    def get_reward(self, state, action):
        return -self.delay[state, action] + self.throughput[state, action]

def main():
    num_routers = 95
    num_links = 38
    learning_rate = 0.17
    discount_factor = 0.95
    epsilon = 0.15
    num_episodes = 1000

    q_router = QRouter(num_routers, num_links, learning_rate, discount_factor)
    network = Network(num_routers, num_links)

    for episode in range(num_episodes):
        state = random.randint(0, num_routers - 1)
        done = False

        while not done:
            action = q_router.choose_action(state, epsilon)
            reward = network.get_reward(state, action)
            next_state = random.randint(0, num_routers - 1)
            q_router.update_q_table(state, action, reward, next_state)
            state = next_state

            if random.random() < 0.1:
                done = True

    print(q_router.q_table)

if __name__ == "__main__":
    main()