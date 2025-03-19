from agent import HumanAgent, QTableAgent, PolicyGradientAgent

agent = HumanAgent()
# agent = QTableAgent("checkpoints/q_table.pkl")
# agent = PolicyGradientAgent('checkpoints/pg.pth')


def get_action(obs):
    return agent.get_action(obs)
