.. -*- mode: rst -*-

====================
Gym-Setpoint Environments
====================

This package contains three custom environments for Gymnasium (formerly OpenAI Gym).

For more information on creating and structuring Python packages, refer to the documentation:
`Documentation <https://github.com/ipython/ipython/tree/master/docs/source>`_

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
    import my_gym_env

    # Create and use MyEnv1
    env1 = gym.make('MyEnv1-v0')
    
    # Create and use MyEnv2
    env2 = gym.make('MyEnv2-v0')
    
    # Create and use MyEnv3
    env3 = gym.make('MyEnv3-v0')

Available Environments
======================

MyEnv1
------

Brief description of MyEnv1.

MyEnv2
------

Brief description of MyEnv2.

MyEnv3
------

Brief description of MyEnv3.

Development
===========

To contribute to this project:

1. Clone the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For more information on Python package structure and best practices, refer to the IPython project structure:
`IPython Project Structure <https://github.com/ipython/ipython>`_

License
=======

Distributed under the MIT License. See `LICENSE` for more information.

Contact
=======

Fabien FURFARO - fabien.furfaro@gmail.com

Project Link: https://github.com/fabienfrfr/Gym-Setpoint
