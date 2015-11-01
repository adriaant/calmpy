# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals, division
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa

logger = logging.getLogger(__name__)


class BaseTool(object):

    def __init__(self, network):
        self.network = network


class ConvergenceMap(BaseTool):
    """Create a convergence map by varying values for indices
       x and y of given input module and plotting winner of
       target module."""

    color_map = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

    def display(self, input_label, x, y, module_name, dim=100, show_plot=True):

        target_mdl = self.network.module_with_name(module_name)
        if not target_mdl:
            return None

        winners = np.empty([dim, dim], dtype='d')
        step = 1.0 / dim
        input_nodes = self.network.inputs[input_label].r

        values = np.arange(step, 1.0 + step, step)
        for i, x_val in enumerate(values):
            for j, y_val in enumerate(values):
                input_nodes[x] = x_val
                input_nodes[y] = y_val
                self.network.test()
                winners[i, j] = target_mdl.winner

        if show_plot:
            cmap = colors.ListedColormap(self.color_map)
            norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)
            plt.imshow(winners, cmap=cmap, norm=norm)
            plt.axis('off')
            plt.show()
        else:
            return winners


class WeightPlot(BaseTool):
    """Create a 3D plot of the weights between two given modules."""

    def display(self, from_name, to_name, show_plot=True, wire=False):
        connection = self.network.get_connection_for(from_name, to_name)
        if connection:
            weights = connection.weights
            dim = len(weights)
            X, Y = np.mgrid[:dim, :dim]
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if wire:
                ax.plot_wireframe(X, Y, weights, rstride=1, cstride=1)
            else:
                ax.plot_surface(X, Y, weights,
                    rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)

            if show_plot:
                plt.show()
            else:
                return plt
        return None


class BifurcationDiagram(BaseTool):
    """Create a bifurcation diagram by varying the value of one input node
       and plotting the value of a target R-node per iteration.
       Assumes input modules have desired pattern set."""

    def display(self, input_mdl, input_idx, mdl_label, r_index, start=0.0, end=1.0, dim=100, iterations=100, transients=10):

        mdl = self.network.module_with_name(mdl_label)
        if not mdl:
            logger.error("Unknown module name!")
            return None

        step = (end - start) / dim
        values = np.arange(start, end + step, step)
        x_vals = np.empty((dim + 1) * iterations, dtype='d')
        y_vals = np.empty((dim + 1) * iterations, dtype='d')

        idx = 0
        for x in values:
            input_mdl.r[input_idx] = x

            self.network.reset()

            # we ignore first couple of iterations since those are transients
            for _ in xrange(0, transients):
                self.network.test_one()

            for _ in xrange(0, iterations):
                self.network.test_one()
                try:
                    x_vals[idx] = x
                    y_vals[idx] = mdl.r[r_index]
                except IndexError:
                    break  # one off end
                idx += 1
            else:
                continue
            break

        plt.scatter(x_vals, y_vals, s=1)
        plt.xlim(start, end)
        plt.ylim(min(y_vals), max(y_vals))
        plt.show()


class SinglePhasePortrait(BaseTool):
    """Create a Poincaré section of a phase portrait by plotting the value of
       a target R-node against that of its paired V-node per iteration.
       Assumes input modules have desired pattern set."""

    def display(self, iterations=500):

        x_vals = np.empty(iterations, dtype='d')
        y_vals = np.empty(iterations, dtype='d')

        self.network.reset()

        # we ignore first 100 iterations since those are transients
        for _ in xrange(0, 100):
            self.network.test_one()

        for idx in xrange(0, iterations):
            self.network.test_one()
            x_vals[idx] = self.network.total_activation()

            self.network.test_one()
            y_vals[idx] = self.network.total_activation()

        color_map = np.arctan2(y_vals, x_vals)
        plt.scatter(x_vals, y_vals, s=42, c=color_map, lw=0)
        edge_r = (max(x_vals) - min(x_vals)) / 100
        edge_v = (max(y_vals) - min(y_vals)) / 100
        plt.xlim(min(x_vals) - edge_r, max(x_vals) + edge_r)
        plt.ylim(min(y_vals) - edge_v, max(y_vals) + edge_v)
        plt.show()


class StackedPhasePortrait(BaseTool):
    """Create multiple Poincaré sections of a phase portrait by plotting the value of
       a target R-node against that of its paired V-node per iteration.
       Assumes input modules have desired pattern set."""

    def display(self, input_mdl, input_idx, width=10, step=0.001, iterations=500):

        step = np.float64(step)
        cur_val = input_mdl.r[input_idx]
        start = cur_val - (width * step)
        if start < 0.0:
            logger.error("Range will be out of bounds!")
            return
        end = cur_val + (width * step)
        if end > 1.0:
            logger.error("Range will be out of bounds!")
            return

        x_vals = np.empty(iterations * (2 * width + 1), dtype='d')
        y_vals = np.empty(iterations * (2 * width + 1), dtype='d')
        values = np.arange(start, end + step, step)

        idx = 0
        for x in values:
            input_mdl.r[input_idx] = x

            self.network.reset()

            # we ignore first 100 iterations since those are transients
            for _ in xrange(0, 100):
                self.network.test_one()

            for _ in xrange(0, iterations):
                self.network.test_one()
                try:
                    x_vals[idx] = self.network.total_activation()
                except IndexError:
                    break

                self.network.test_one()
                try:
                    y_vals[idx] = self.network.total_activation()
                except IndexError:
                    break

                print "{0}: {1}".format(x_vals[idx], y_vals[idx])
                idx += 1
            else:
                continue
            break

        color_map = np.arctan2(y_vals, x_vals)
        plt.scatter(x_vals, y_vals, s=42, c=color_map, lw=0)
        edge_r = (max(x_vals) - min(x_vals)) / 10
        edge_v = (max(y_vals) - min(y_vals)) / 10
        plt.xlim(min(x_vals) - edge_r, max(x_vals) + edge_r)
        plt.ylim(min(y_vals) - edge_v, max(y_vals) + edge_v)
        plt.show()
