# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from django.utils import six

from simulator import load_network

logger = logging.getLogger(__name__)


def train_with_activation_display(network_name, mdl_name):
    """Trains network while displaying node activations of given module."""

    def train(self, epochs=100, iterations=50):
        """Trains the network for the given amount of epochs and iterations."""

        input_mdl = six.next(six.itervalues(self.inputs))
        patterns = np.random.uniform(0.0, 1.0, size=(epochs, 2))

        for pat in patterns:
            # reset activations
            for mdl in self.modules:
                mdl.reset()

            # set pattern
            input_mdl.r = pat

            # activation flow and weight update
            for _ in xrange(0, iterations):
                # update activations
                for mdl in self.modules:
                    mdl.activate()

                # update weights
                for mdl in self.modules:
                    mdl.change_weights()

                # swap acts
                for mdl in self.modules:
                    mdl.swap_activations()

    network = load_network(network_name)
    mdl = network.module_with_name(mdl_name)
    input_mdl = six.next(six.itervalues(network.inputs))
    input_mdl.r = np.array([1.0, 0.0], dtype='d')
    for mdl in network.modules:
        mdl.reset()

    # set up node display
    fig = plt.figure()
    plt.suptitle(str(input_mdl.r))
    ax = plt.axes(xlim=(0, 5), ylim=(0, 3))
    r_1 = plt.Rectangle((1, 0), 1, 0.0, fc='r')
    ax.add_patch(r_1)

    r_2 = plt.Rectangle((3, 0), 1, 0.0, fc='r')
    ax.add_patch(r_2)

    v_1 = plt.Rectangle((1, 1.5), 1, 0.0, fc='b')
    ax.add_patch(v_1)

    v_2 = plt.Rectangle((3, 1.5), 1, 0.0, fc='b')
    ax.add_patch(v_2)

    def animate(i):
        # update activations
        for mdl in network.modules:
            mdl.activate()

        # swap acts
        for mdl in network.modules:
            mdl.swap_activations()

        # update weights
        for mdl in network.modules:
            mdl.change_weights()

        r_1.set_height(mdl.r[0])
        r_2.set_height(mdl.r[1])
        v_1.set_height(mdl.v[0])
        v_2.set_height(mdl.v[1])

    anim = animation.FuncAnimation(fig, animate,
                                   frames=50,
                                   interval=20,
                                   blit=False)
    anim.save('/tmp/pat1.mp4', fps=30, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])

    # reset activations
    for mdl in network.modules:
        mdl.reset()
    input_mdl.r = np.array([0.0, 1.0], dtype='d')
    plt.suptitle(str(input_mdl.r))

    anim = animation.FuncAnimation(fig, animate,
                                   frames=50,
                                   interval=20,
                                   blit=False)
    anim.save('/tmp/pat2.mp4', fps=30, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
    # plt.show()

    return network
