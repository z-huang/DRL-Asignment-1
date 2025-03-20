from agent import PolicyGradientAgent

agent = PolicyGradientAgent('checkpoints/pg.pth')


def get_action(obs):
    return agent.get_action(obs)
