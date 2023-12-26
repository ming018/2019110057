import numpy as np

# UAV의 상태와 행동, 보상을 정의하는 클래스
class DisasterResponseModel:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_transition_matrix = np.zeros((num_states, num_actions, num_states))
        self.reward_matrix = np.random.rand(num_states, num_actions)  # 임의로 설정된 보상 값

    def set_transitions(self, state, action, next_state, prob):
        self.state_transition_matrix[state, action, next_state] = prob

    def step(self, state, action):
        next_state_probs = self.state_transition_matrix[state, action]
        next_state = np.random.choice(self.num_states, 1, p=next_state_probs)[0]
        reward = self.reward_matrix[state, action]
        return next_state, reward

# 정책 반복 알고리즘 구현
def policy_iteration(model, discount_factor=0.95, max_iterations=1000):
    policy = np.zeros(model.num_states, dtype=int)
    value_function = np.zeros(model.num_states)

    for _ in range(max_iterations):
        # 정책 평가
        while True:
            new_value_function = np.copy(value_function)
            for state in range(model.num_states):
                action = policy[state]
                value_function[state] = sum([trans_prob * (model.reward_matrix[state, action] + discount_factor * new_value_function[next_state])
                                             for next_state, trans_prob in enumerate(model.state_transition_matrix[state, action])])
            if np.max(np.abs(new_value_function - value_function)) < 1e-4:
                break

        # 정책 개선
        policy_stable = True
        for state in range(model.num_states):
            old_action = policy[state]
            q_values = np.zeros(model.num_actions)
            for action in range(model.num_actions):
                q_values[action] = sum([trans_prob * (model.reward_matrix[state, action] + discount_factor * value_function[next_state])
                                        for next_state, trans_prob in enumerate(model.state_transition_matrix[state, action])])
            best_action = np.argmax(q_values)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, value_function

# 예제 모델 생성 및 정책 반복 실행
num_states = 10  # 가정된 상태의 수
num_actions = 4  # 가정된 행동의 수
model = DisasterResponseModel(num_states, num_actions)

# 임시 상태 전이 확률 설정
for state in range(num_states):
    for action in range(num_actions):
        next_state = (state + action) % num_states
        model.set_transitions(state, action, next_state, 1.0)  # 단순화를 위해 결정론적 전이

# 정책 반복 실행
optimal_policy, value_function = policy_iteration(model)
print("최적 정책:", optimal_policy)
print("가치 함수:", value_function)
