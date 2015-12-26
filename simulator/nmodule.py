# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101,W0201,W1202,W0622
from __future__ import absolute_import, unicode_literals, division
import logging
from math import floor
import numpy as np
from django.utils.six.moves import xrange
from django.utils.encoding import python_2_unicode_compatible
from .exceptions import CALMException
from .helpers import printoptions
from .utils import random_val

logger = logging.getLogger(__name__)
p_0 = np.float64(0)
p_1 = np.float64(1)


class ModuleFactory(object):
    """
    Factory to dynamically instantiate CALM module objects based on given type.
    """

    @classmethod
    def build(cls, obj_type, *args, **kwargs):

        def _get_object_for_type(obj_type, subclasses):
            for c in subclasses:
                if obj_type in c.__name__.lower():
                    return c(*args, **kwargs)
                obj = _get_object_for_type(obj_type, c.__subclasses__())
                if obj:
                    return obj
            return None

        obj_type = obj_type.lower()
        return _get_object_for_type(obj_type, cls.__subclasses__())


@python_2_unicode_compatible
class InputModule(ModuleFactory):
    """
    Defines an input module.
    """

    def __init__(self, name, size, parameters):
        self.size = size
        self.name = name
        self.r = np.zeros(self.size, dtype='d')

    def __str__(self):
        return self.name


@python_2_unicode_compatible
class StandardModule(ModuleFactory):
    """
    Defines a CALM module of a given type.
    """

    def __init__(self, name, size, parameters):
        self.size = size
        self.name = name
        self.parameters = parameters
        self.connections = []
        self.init_nodes()
        self.init_intraweights()
        self.define_vectorized_funcs()
        self.decay_value = np.float64((1.0 - self.parameters['K_A']))
        self.random_func = random_val
        self.winner = 0

    def define_vectorized_funcs(self):
        """Create some functions for fast vectorized computation."""

        def update_activation(a, b):
            """CALM activation update function."""
            current_act = np.float64(a)
            new_act = np.float64(b)
            decay = self.decay_value * current_act
            if new_act >= 0.0:
                return decay + (new_act / (p_1 + new_act)) * (p_1 - decay)
            return decay + (new_act / (p_1 - new_act)) * decay

        self.update_func = np.vectorize(update_activation, otypes=[np.float64])

    def init_nodes(self):
        """Initialize arrays for the four types of nodes."""
        self.r = np.zeros(self.size, dtype='d')
        self.v = np.zeros(self.size, dtype='d')
        self.a = np.zeros(1, dtype='d')
        self.e = np.zeros(1, dtype='d')
        self.r_new = None
        self.v_new = None
        self.a_new = None
        self.e_new = None

    def init_v_weights(self):
        """Initialize connections to R nodes from V"""
        self.v_to_r = np.full((self.size, self.size), self.parameters['CROSS'], dtype='d')
        np.fill_diagonal(self.v_to_r, self.parameters['DOWN'])

    def init_intraweights(self):
        """Initialize matrices for all intra-weights to be used for
        quick calculation of incoming activations."""

        # connections to R nodes from V
        self.init_v_weights()

        # connections to V from R and other V-nodes
        r_to_v = np.full((self.size, self.size), 0, dtype='d')
        np.fill_diagonal(r_to_v, self.parameters['UP'])
        v_to_v = np.full((self.size, self.size), self.parameters['FLAT'], dtype='d')
        np.fill_diagonal(v_to_v, 0)
        self.rv_to_v = np.hstack((r_to_v, v_to_v))

        # connections to A-node
        self.r_to_a = np.full((1, self.size), self.parameters['LOW'], dtype='d')
        self.v_to_a = np.full((1, self.size), self.parameters['HIGH'], dtype='d')

    def connect_from(self, from_mdl):
        self.connections.append(CALMConnection(from_mdl, self, self.parameters))

    def check_convergence(self):
        """If one node is larger than 0.9 we have a winner.
           Note that winners use 1-based indexing."""

        # first check if we have one R-node with high activation
        winners = (self.r >= self.parameters['HIGHCRIT'])
        if winners.sum() == 1:
            self.winner = np.where(winners == True)[0][0] + 1  # noqa
            return True

        # possibly input is low, so check if other nodes are below threshold
        losers = (self.r < self.parameters['LOWCRIT'])
        if self.size - losers.sum() == 1:
            self.winner = np.where(losers == False)[0][0] + 1 # noqa
            return True
        return False

    def reset(self, soft=True):
        """Resets activations and optionally weights."""
        self.winner = 0
        for node_type in ('r', 'v', 'a', 'e'):
            node = getattr(self, node_type)
            node.fill(0.0)

        if not soft:
            for connection in self.connections:
                connection.reset()

    def activate(self, testing=False):
        """Calculates incoming activations to all nodes."""
        self.activate_r(testing)
        self.activate_internal()

    def activate_r(self, testing):
        """
            Calculate the new activation of R-nodes.
            When 'testing' is True, then no random pulses from E-node are sent.
        """
        # activations from connected modules
        r_acts = self.connections[0].from_module.r
        r_weights = self.connections[0].weights
        for i in range(len(self.connections) - 1):
            connection = self.connections[i + 1]
            r_acts = np.append(r_acts, connection.from_module.r)
            r_weights = np.hstack((r_weights, connection.weights))
        new_acts = np.dot(r_weights, r_acts)

        # activations from v-nodes
        tmp = np.dot(self.v_to_r, self.v)
        new_acts += tmp

        if not testing:
            # Create random pulses based on E-node activation
            e_to_r = self.parameters['ER'] * self.random_func(self.e[0], self.size)
            new_acts += np.dot(e_to_r, np.ones(1, dtype='d'))

        self.r_new = self.update_func(self.r, new_acts)

    def activate_internal(self):
        """Calculate activation of V-, A- and E-nodes."""
        # V-activations: stack R and V activations
        r_and_v_acts = np.append(self.r, self.v)
        self.v_new = self.update_func(self.v, np.dot(self.rv_to_v, r_and_v_acts))

        # A activations
        sumr = np.sum(self.r) * self.parameters['LOW']
        sumv = np.sum(self.v) * self.parameters['HIGH']
        self.a_new = self.update_func(self.a, sumr + sumv)

        # E activations
        self.e_new = self.update_func(self.e, self.parameters['AE'] * self.a)

    def change_weights(self):
        """Apply the learning rule."""

        # classic
        # mu = self.parameters['D_L'] + self.parameters['WMUE_L'] * self.e

        # gaussian
        e_val = (self.e - self.parameters['G_L']) * (self.e - self.parameters['G_L'])
        mu = self.parameters['D_L'] + self.parameters['WMUE_L'] * np.exp(p_0 - (e_val / self.parameters['G_W']))

        for i, act_i in np.ndenumerate(self.r):
            # build up sum of background activation
            back_act = p_0
            for connection in self.connections:
                for j in xrange(0, len(connection.from_module.r)):
                    back_act += connection.weights[i][j] * connection.from_module.r[j]
            for connection in self.connections:
                connection.change_weight(i, act_i, mu, back_act)

    def swap_activations(self):
        """Swap new activations to current."""
        self.r = self.r_new
        self.v = self.v_new
        self.a = self.a_new
        self.e = self.e_new

    def get_connection_from(self, from_mdl):
        for connection in self.connections:
            if connection.from_module == from_mdl:
                return connection
        return None

    def display_weights(self):
        for connection in self.connections:
            connection.display_weights()

    def get_weights(self):
        """Used by CALMNetwork::save_weights."""
        connection_data = {}
        for connection in self.connections:
            connection_data[connection.from_module.name] = connection.weights
        return connection_data

    def set_weights(self, from_mdl, weight_data):
        """Used by CALMNetwork::load_weights."""
        for connection in self.connections:
            if connection.from_module == from_mdl:
                connection.weights = weight_data
                return

    def display_activations(self):
        s = "V: "
        for node in self.v:
            s += "{:10.6f}".format(node)
        s += " A: {:10.6f}".format(self.a[0])
        logger.info(s)
        s = "R: "
        for node in self.r:
            s += "{:10.6f}".format(node)
        s += " E: {:10.6f}".format(self.e[0])
        logger.info(s)

    def __str__(self):
        return self.name


class MapModule(StandardModule):
    """
    Defines a CALM module of a given type.
    """

    def init_v_weights(self):
        """Initialize connections to R nodes from V with values
           dependent on distance between R and V node pair."""

        if self.size % 2 == 0:
            mdl_size = self.size
        else:
            mdl_size = self.size + 1
        self.v_to_r = np.empty((self.size, self.size), dtype='d')
        middle = int(floor(mdl_size / 2.0))
        n = np.float64(mdl_size)
        # calculate optimal sigma
        sigma = (-4.0 / n) * np.log((0.01 + np.exp(-0.25 * n)) / (n + 1.0))

        v_weights = []
        for i in range(0, middle + 1):
            v_weights.append((n + 1.0) * np.exp(0.0 - (sigma * i * i) / n) - n - 1.0 + self.parameters['DOWN'])

        for i in xrange(0, self.size):
            for j in xrange(0, self.size):
                dist = abs(i - j)  # distance between R and V node
                if dist > middle:
                    dist = mdl_size - dist  # correct for size

                # SIGMA depends on module size. A module size up to 20 has
                # experimentally been defined to have an optimal sigma around 0.06.
                # With more nodes, this sigma slightly increases.
                # With 64 nodes, sigma should be picked around 0.15
                # see https://www.dropbox.com/s/8d0u9o71sn4sbrh/gaussian.pdf
                self.v_to_r[i, j] = v_weights[dist]

        with printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True):
            logger.info("Sigma: {0}\nV to R weights: {1}".format(sigma, self.v_to_r))


@python_2_unicode_compatible
class CALMConnection(object):
    """
    Defines a connection between two modules.
    """

    def __init__(self, from_mdl, to_mdl, parameters):
        self.parameters = parameters
        if isinstance(to_mdl, InputModule):
            raise CALMException(detail="Input modules cannot receive connections!")
        self.to_module = to_mdl
        self.from_module = from_mdl
        self.weights = np.full((to_mdl.size, from_mdl.size), self.parameters['INITWT'], dtype='d')

    def reset(self):
        self.weights.fill(self.parameters['INITWT'])

    def change_weight(self, to_idx, act_to_idx, mu, back_acts):
        """Calculate the change in weight"""

        for j, act_j in np.ndenumerate(self.from_module.r):
            w = self.weights[to_idx, j]
            dw = mu * act_to_idx * (
                (self.parameters['K_Lmax'] - w) * act_j -
                self.parameters['L_L'] * (w - self.parameters['K_Lmin']) * (back_acts - w * act_j))

            self.weights[to_idx, j] += dw

        # restrict weights to mParameters[K_Lmax], mParameters[K_Lmin]
        self.weights[self.weights < self.parameters['K_Lmin']] = self.parameters['K_Lmin']
        self.weights[self.weights > self.parameters['K_Lmax']] = self.parameters['K_Lmax']

    def display_weights(self):
        logger.info("    {0}➞{1}:".format(self.from_module.name, self.to_module.name))
        for j in xrange(0, len(self.from_module.r)):
            s = "    "
            for i in xrange(0, len(self.to_module.r)):
                s += "{:10.6f}".format(self.weights[i, j])
            logger.info(s)

    def __str__(self):
        return "{0}➞{1}".format(self.from_module.name, self.to_module.name)
