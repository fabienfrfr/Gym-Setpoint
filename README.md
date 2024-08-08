.. -*- mode: rst -*-

====================
Gym-Setpoint Environments
====================

This package contains three custom environments based on python-control for Gymnasium (formerly OpenAI Gym).

For more information on creating and structuring Python packages, refer to the documentation:
`Documentation <https://github.com/fabienfrfr/Gym-Setpoint/blob/main/doc/SetPoint_Learning.pdf>`_

Citation
========

If you use this package in your research or project, please cite it as follows:

.. code-block:: bibtex

    @misc{gym_setpoint,
        author = {Fabien Furfaro},
        title = {Gym-Setpoint Environments},
        year = {2024},
        publisher = {GitHub},
        journal = {GitHub repository},
        url = {https://github.com/fabienfrfr/Gym-Setpoint}
    }

Installation
============

To install this package, run the following command:

.. code-block:: bash

    pip install git+https://github.com/fabienfrfr/Gym-Setpoint@main

Usage
=====

After installation, you can use the environments as follows:

.. code-block:: python

    import gymnasium as gym
    import gym_setpoint
    # custom import
    from gym_setpoint.envs import gym_wrap, lti_env, multi_lti

    # Create and use Gym-Wrap
    env1 = gym.make('GymWrap-v0')
    # or customised import
    env1 = gym_wrap.GymWrap(config = {
                     "env":'CartPole-v1',
                     "mode":2,
                     "classic":False,
                     "dim":None,
                     "is_discrete":False,
                     "N_space":3})
    
    # Create and use LTI-Env
    env2 = gym.make('LtiEnv-v0')
    env2 = lti_env.LtiEnv(config={
                  "env_mode":2,
                  "update_setpoint":True,
                  "reset_X_start":True,
                  "tf":None,
                  "reset":True,
                  "isdiscrete": False,
                  "SpaceState":None,
                  "setpoint":None,
                  "env_config":None,
                  "modular":False,
                  "return_action":True,
                  "return_speed":False,
                  "order":3,
                  "t":10,
                  "N":250})
    
    # Create and use Multi-LTI
    env3 = gym.make('MultiLti-v0')
    env3 = multi_lti.MultiLti(config={
                  "env_mode":None,
                  "reset":True,
                  "n":32,
                  "t":10,
                  "N":250,})

Available Environments
======================

Gym-Wrap
------

"Gym control" environment wrapper, transforms any non-linear controllable environment, into a linearized version. The state space is reduced to a single dimension, involving several hidden variables, which can make the environment more difficult to control, but whose input variables of the learning models are always the same.

LTI-Env
------

Controllable linear state space models of order N, with a single input and a single output. The output is then modified following the setpoint approach and adapted to reinforcement learning.


Multi-LTI
------

Linear controllable state space models of order N, with M inputs and M outputs, interconnected or not. The output is also modified following the setpoint approach and adapted to reinforcement learning. Here, the output is adapted for a control with an output with multiple actions and a single rewards, but the multi-agent approach is also proposed.


Development
===========

This project was conducted as part of research at Capgemini Engineering in France.

License
=======

Distributed under the MIT License. See `LICENSE` for more information.

Contact
=======

`Fabien FURFARO <https://www.linkedin.com/in/fabien-furfaro>`_

