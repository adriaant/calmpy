# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals, division
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from django.utils import six

from simulator import load_network

logger = logging.getLogger(__name__)

ignore_first_animation = True  # weird matlibplot bug where animation index 0 is sent twice
num_presentations = 50
num_iterations = 50
current_presentation = 0


def train_with_activation_display(network_name, mdl_name):
    """Trains network while displaying node activations of given module."""

    network = load_network(network_name)
    for cur_mdl in network.modules:
        cur_mdl.reset()

    mdl = network.module_with_name(mdl_name)

    # pick first input module (this code won't work with multi-input modules)
    input_mdl = six.next(six.itervalues(network.inputs))

    num_frames = len(network.patterns) * num_iterations * num_presentations

    # set up node display
    fig = plt.figure()

    num_nodes = max(len(input_mdl.r), len(mdl.r)) + 1
    ax = plt.axes(xlim=(0, 0.5 + num_nodes), ylim=(0, 3.5), frameon=True)
    plt.tick_params(
        axis='both',
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    input_nodes = []
    x = 0.5
    for node in input_mdl.r:
        patch = plt.Rectangle((x, 0), 0.5, 0.0, fc='k')
        ax.add_patch(patch)
        input_nodes.append(patch)
        x += 1.0

    r_nodes = []
    x = 0.5
    for node in mdl.r:
        patch = plt.Rectangle((x, 1), 0.5, 0.0, fc='r')
        ax.add_patch(patch)
        r_nodes.append(patch)
        x += 1.0

    e = plt.Rectangle((x, 1), 0.5, 0.0, fc='y')
    ax.add_patch(e)

    v_nodes = []
    x = 0.5
    for node in mdl.v:
        patch = plt.Rectangle((x, 2.5), 0.5, 0.0, fc='b')
        ax.add_patch(patch)
        v_nodes.append(patch)
        x += 1.0

    a = plt.Rectangle((x, 2.5), 0.5, 0.0, fc='g')
    ax.add_patch(a)

    def learn_animate(i):
        print("animation index: {0}".format(i))

        global ignore_first_animation
        if ignore_first_animation:
            ignore_first_animation = False
            return

        global current_presentation, num_iterations

        if i % num_iterations == 0:
            for cur_mdl in network.modules:
                cur_mdl.reset()

            pat = network.patterns[current_presentation]
            input_mdl.r = pat[input_mdl.name]
            for idx, val in enumerate(input_mdl.r):
                input_nodes[idx].set_height(val / 2.0)
            current_presentation += 1
            if current_presentation >= len(network.patterns):
                current_presentation = 0

        # update activations
        for cur_mdl in network.modules:
            cur_mdl.activate()

        # swap acts
        for cur_mdl in network.modules:
            cur_mdl.swap_activations()

        # update weights
        for cur_mdl in network.modules:
            cur_mdl.change_weights()

        for idx, val in enumerate(mdl.r):
            r_nodes[idx].set_height(val)
        for idx, val in enumerate(mdl.v):
            v_nodes[idx].set_height(val)

        a.set_height(mdl.a[0])
        e.set_height(mdl.e[0])

    anim = animation.FuncAnimation(fig, learn_animate,
                                   frames=num_frames,
                                   interval=20,
                                   blit=False,
                                   repeat=False)
    anim.save("/tmp/{0}_learning.mp4".format(network.name), fps=25, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])

    def test_animate(i):
        print("animation index: {0}".format(i))

        global ignore_first_animation
        if ignore_first_animation:
            ignore_first_animation = False
            return

        global current_presentation, num_iterations

        if i % num_iterations == 0:
            for cur_mdl in network.modules:
                cur_mdl.reset()

            pat = network.patterns[current_presentation]
            input_mdl.r = pat[input_mdl.name]
            for idx, val in enumerate(input_mdl.r):
                input_nodes[idx].set_height(val / 2.0)
            current_presentation += 1
            if current_presentation >= len(network.patterns):
                current_presentation = 0

        # update activations
        for cur_mdl in network.modules:
            cur_mdl.activate(testing=True)

        # swap acts
        for cur_mdl in network.modules:
            cur_mdl.swap_activations()

        for idx, val in enumerate(mdl.r):
            r_nodes[idx].set_height(val)
        for idx, val in enumerate(mdl.v):
            v_nodes[idx].set_height(val)

        a.set_height(mdl.a[0])
        e.set_height(mdl.e[0])

    global current_presentation
    global ignore_first_animation
    current_presentation = 0
    ignore_first_animation = True
    num_frames = len(network.patterns) * num_iterations
    anim = animation.FuncAnimation(fig, test_animate,
                                   frames=num_frames,
                                   interval=20,
                                   blit=False)
    anim.save("/tmp/{0}_testing.mp4".format(network.name), fps=25, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
    # plt.show()

    return network
