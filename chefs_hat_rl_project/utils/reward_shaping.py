def sparse_reward(original_reward):
    return original_reward


def shaped_reward(original_reward, info):
    reward = original_reward

    if isinstance(info, dict):
        if info.get("won_round", False):
            reward += 0.1

        if info.get("lost_cards", 0) > 0:
            reward -= 0.01 * info["lost_cards"]

    return reward