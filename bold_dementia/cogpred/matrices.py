import math
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import seaborn as sns

from neuroginius.networks import group_by_networks
from neuroginius.atlas import Atlas
from neuroginius.iterables import unique
import itertools as it

def default_agg_func(block):
    return (block.mean(),)

class MatrixResult:
    def __init__(self, matrix, atlas) -> None:
        self.atlas = atlas
        self.matrix = matrix
        self._set_sorted_matrix()
    
    def _set_sorted_matrix(self):
        """Reorganize the matrix by macro labels, store
        the sorted matrix and a mapping from networks name
        to indexes in the sorted matrix
        """

        ticks, sort_index = group_by_networks(self.atlas.macro_labels)
        matrix_sort = np.ix_(sort_index, sort_index)

        self.sorted_matrix = self.matrix[matrix_sort]
        new_labels = sorted(tuple(unique(self.atlas.macro_labels)))
        self.network_to_idx = pd.Series(dict(zip(
            new_labels,
            it.pairwise(ticks)
        )))

    def get_macro_matrix(self, agg_func=default_agg_func):
        """Get a matrix reorganized by networks

        Args:
            agg_func (function, optional): function to compute
            the aggregated of each cell, from the block of original
            values. Defaults to default_agg_func, which performs a simple
            mean

        Returns:
            DataFrame: summary per network of the original matrix.
        """
        gen = self._gen_macro_values(
            agg_func=agg_func
        )
        comparisons = pd.DataFrame(gen, columns=["node_a", "node_b", "connectivity"])
        pivoted = comparisons.pivot(index="node_a", columns="node_b")
        return pivoted.loc[:, "connectivity"]

    # This could be a function on its own
    def _gen_macro_values(self, agg_func):
        for network_a, network_b in it.product(self.network_to_idx.index, self.network_to_idx.index):
            loc_a, loc_b = self.network_to_idx[network_a], self.network_to_idx[network_b]
            block = self.matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]]

            yield network_a, network_b, *agg_func(block)
    
    def plot(self):
        pass
        
    
        

def compute_mat_size(l, with_diag=False):
    # Mat size is the positive root of :
    # n**2 - b*n - 2l = 0 
    # Where l is the length of pvalues array
    # and n is the square matrix size.
    if with_diag:
        b = 1
    else:
        b = -1
    n = (-b + math.sqrt(1 + 8 * l)) / 2
    if n != int(n):
        raise ValueError(f"Array of lenght {l} cannot be reshaped as a square matrix\
                         (with_diag is {with_diag})")
    return int(n)

def plot_matrix(
    mat, atlas, macro_labels=True, bounds=None, axes=None, **sns_kwargs
):
    """Simplified version of the plot_matrices function. Only displays
    a single matrix.

    Args:
        mat (_type_): _description_
        atlas (Bunch): sklearn bunch containing labels and
        macro labels id macro_labels is True
        macro_labels (bool, optional): _description_. Defaults to True.
        bounds (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "seismic".

    """
    mat = mat.copy()
    n_regions = mat.shape[0]
    mat[list(range(n_regions)), list(range(n_regions))] = 0

    # In general we want a colormap that is symmetric around 0
    span = max(abs(mat.min()), abs(mat.max()))
    if bounds is None:
        bounds = (-span, span)

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    if macro_labels:
        networks = np.array(atlas.macro_labels)

        sort_index = np.argsort(networks)
        ticks = []
        lbls = []
        prev_label = None
        for i, label in enumerate(networks[sort_index]):
            if label != prev_label:
                ticks.append(i)
                lbls.append(label)
                prev_label = label
                axes.hlines(i, 0, n_regions, colors="black", linestyles="dotted", linewidths=2)
                axes.vlines(i, 0, n_regions, colors="black", linestyles="dotted", linewidths=2)

        ticks.append(i + 1)

    else:
        sort_index = np.arange(n_regions)

    sns.heatmap(
        mat[np.ix_(sort_index, sort_index)],
        ax=axes,
        vmin=bounds[0],
        vmax=bounds[1],
        **sns_kwargs
    )

    if macro_labels:
        axes.yaxis.set_minor_locator(FixedLocator(ticks))
        axes.yaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        axes.xaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        axes.set_yticklabels(lbls, rotation=0)
        axes.set_xticklabels(lbls, rotation=30)

    return axes