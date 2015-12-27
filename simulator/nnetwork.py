# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101,W0201,W1202,W0622
from __future__ import absolute_import, unicode_literals, division
import os
import io
import json
import logging
import numpy as np
from django.utils import six
from django.utils.six.moves import xrange
from django.utils.encoding import python_2_unicode_compatible
from .exceptions import CALMException
from .helpers import printoptions
from .utils import ensure_directory, NumpyEncoder, json_numpy_obj_hook
from .nmodule import ModuleFactory, InputModule

logger = logging.getLogger(__name__)


@python_2_unicode_compatible
class CALMNetwork(object):
    """
    Defines a CALM network with variable number of modules and a set of parameters.
    """

    original_calm = False  # Use if you want activation swaps after weight updates
    allow_dynamic_resizing = False  # Use if modules should grow/shrink
    resize_check_interval = 5  # Number of epochs between dynamic resizing

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.inputs = {}
        self.modules = []
        self.patterns = None
        self.data_dir = ensure_directory(name)

    def build_from_definition(self, definition):
        """Builds up a CALM network based on JointJS graph data."""

        def _normalize_type(s):
            return s.split('.')[1].lower()

        cells = definition['cells']
        links = []
        modules = {}
        # define modules first
        for cell in cells:
            if cell['type'] == 'link':
                links.append(cell)
            else:
                mdl_id = cell['id']
                mdl_name = cell['name']
                mdl_size = int(cell['mdl_size'])
                mdl_type = _normalize_type(cell['type'])
                modules[mdl_id] = ModuleFactory.build(mdl_type, mdl_name, mdl_size, self.parameters)

        # connect the modules
        for link in links:
            from_mdl = link['source']['id']
            to_mdl = link['target']['id']
            mdl = modules[to_mdl]
            mdl.connect_from(modules[from_mdl])

        # sort out the modules
        for mdl in modules.values():
            if isinstance(mdl, InputModule):
                self.inputs[mdl.name] = mdl
            else:
                self.modules.append(mdl)
        del modules

    def load_patterns(self, name='patterns.json'):
        """Loads patterns from a JSON file."""
        patterns_path = os.path.join(self.data_dir, name)
        if not os.path.exists(patterns_path):
            raise CALMException(detail='No patterns file found!')
        try:
            with io.open(patterns_path) as patterns_file:
                self.patterns = json.loads(patterns_file.read())
        except:
            err_msg = 'Could not load patterns'
            logger.exception(err_msg)
            raise CALMException(detail=err_msg)

        # convert text to arrays
        for pattern in self.patterns:
            for (k, v) in six.iteritems(pattern):
                pattern[k] = np.fromstring(v, dtype='d', sep=' ')

    def load_weights(self, name):
        """Loads weights from file with given name, located in data directory."""
        weight_file = os.path.join(self.data_dir, name)
        with open(weight_file) as f:
            weight_data = json.load(f, object_hook=json_numpy_obj_hook)
        if weight_data:
            for (to_mdl_name, conns) in six.iteritems(weight_data):
                to_mdl = self.module_with_name(to_mdl_name)
                for (from_mdl_name, weights) in six.iteritems(conns):
                    from_mdl = self.module_with_name(from_mdl_name)
                    to_mdl.set_weights(from_mdl, weights)

    def save_weights(self, name):
        """Saves weights to file with given name, located in data directory."""
        weight_file = os.path.join(self.data_dir, name)
        weight_data = {}
        for mdl in self.modules:
            weight_data[mdl.name] = mdl.get_weights()

        with open(weight_file, 'w') as f:
            json.dump(weight_data, f, cls=NumpyEncoder)

    def train(self, epochs=100, iterations=50, reset=False):
        """Trains the network for the given amount of epochs and iterations."""

        if reset:
            # initialize weights first
            self.reset(False)

        for epoch in xrange(0, epochs):
            permuted_pattern_set = np.random.permutation(self.patterns)
            for pat in permuted_pattern_set:

                # reset activations
                for mdl in self.modules:
                    mdl.reset()

                # set pattern
                for (k, v) in six.iteritems(pat):
                    self.inputs[k].r = v

                # activation flow and weight update
                for _ in xrange(0, iterations):
                    # update activations
                    for mdl in self.modules:
                        mdl.activate()

                    if not self.original_calm:
                        # swap acts
                        for mdl in self.modules:
                            mdl.swap_activations()

                    # update weights
                    for mdl in self.modules:
                        mdl.change_weights()

                    if self.original_calm:
                        # swap acts
                        for mdl in self.modules:
                            mdl.swap_activations()

            if self.allow_dynamic_resizing and epoch % self.resize_check_interval == 0:
                self.resize_modules()

    def reset(self, soft=True):
        """Resets activations and optionally weights"""
        for mdl in self.modules:
            mdl.reset(soft)

    def resize_modules(self):
        """Dynamically resizes modules if necessary based on potentials."""
        for mdl in self.modules:
            mdl.resize()

    def performance_check(self, iterations=100):
        """Tests the network on each pattern and reports activations and winners."""
        for pat in self.patterns:
            with printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True):
                for (k, v) in six.iteritems(pat):
                    logger.info("Pattern: {0}: {1}".format(k, v))

            # reset activations
            for mdl in self.modules:
                mdl.reset()

            # set pattern
            for (k, v) in six.iteritems(pat):
                self.inputs[k].r = v

            # activation flow
            for _ in xrange(0, iterations):
                # update activations
                for mdl in self.modules:
                    mdl.activate()
                for mdl in self.modules:
                    mdl.swap_activations()
            self.check_convergence()

            with printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True):
                for mdl in self.modules:
                    logger.info("Module {0}: {1} => {2}".format(mdl.name, mdl.r, mdl.winner))

    def test(self, iterations=100):
        """Tests the network for the given amount of epochs and iterations."""

        self.reset()

        for _ in xrange(0, iterations):
            self.test_one()

            # determine winners
            if self.check_convergence():
                return

    def test_one(self):
        """Performs one update cycle"""
        # update activations
        for mdl in self.modules:
            mdl.activate(testing=True)

        # swap acts
        for mdl in self.modules:
            mdl.swap_activations()

    def check_convergence(self):
        """Determines winners in modules. Returns true if all
           modules have winners."""
        for mdl in self.modules:
            if not mdl.check_convergence():
                return False
        return True

    def total_activation(self):
        activation_sum = np.float64(0.0)
        for mdl in self.modules:
            activation_sum += np.sum(mdl.r)
            activation_sum += np.sum(mdl.v)
        return activation_sum

    def module_with_name(self, name):
        if name in self.inputs:
            return self.inputs[name]

        for mdl in self.modules:
            if mdl.name == name:
                return mdl
        return None

    def get_connection_for(self, from_label, to_label):
        to_mdl = self.module_with_name(to_label)
        from_mdl = self.module_with_name(from_label)
        if to_mdl and from_mdl:
            return to_mdl.get_connection_from(from_mdl)

    def display(self):
        """Prints out the network"""
        logger.info("Name: {0}".format(self.name))
        logger.info("Input modules (size):")
        for mdl in self.inputs.values():
            logger.info("    {0} ({1})".format(mdl.name, mdl.size))

        logger.info("Modules (size):")
        for mdl in self.modules:
            logger.info("    {0} ({1})".format(mdl.name, mdl.size))

        logger.info("Connections:")
        for mdl in self.modules:
            mdl.display_weights()

    def __str__(self):
        return self.name
