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

    # Create and use Gym-Wrap
    env1 = gym.make('GymWrap-v0')
    
    # Create and use LTI-Env
    env2 = gym.make('LtiEnv-v0')
    
    # Create and use Multi-LTI
    env3 = gym.make('MultiLti-v0')

Available Environments
======================

Gym-Wrap
------

Brief description of Gym-Wrap.

LTI-Env
------

Brief description of LTI-Env.

Multi-LTI
------

Brief description of Multi-LTI.

Development
===========

This project was conducted as part of research at Capgemini Engineering in France.

License
=======

Distributed under the MIT License. See `LICENSE` for more information.

Contact
=======

Fabien FURFARO - fabien.furfaro@gmail.com

Project Link: https://github.com/fabienfrfr/Gym-Setpoint
