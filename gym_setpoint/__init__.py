from gymnasium.envs.registration import register

register(
    id='GymWrap-v0',
    entry_point='gym_setpoint.envs:gym_wrap',
)

register(
    id='LtiEnv-v0',
    entry_point='gym_setpoint.envs:lti_env',
)

register(
    id='MultiLti-v0',
    entry_point='gym_setpoint.envs:multi_lti',
)
