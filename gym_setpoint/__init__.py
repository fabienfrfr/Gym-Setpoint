from gymnasium.envs.registration import register

register(
    id='GymWrap-v0',
    entry_point='gym_setpoint.envs:GymWrap',
)

register(
    id='LtiEnv-v0',
    entry_point='gym_setpoint.envs:LtiEnv',
)

register(
    id='MultiLti-v0',
    entry_point='gym_setpoint.envs:MultiLti',
)
