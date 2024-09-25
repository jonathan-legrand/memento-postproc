import math
import itertools as it
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome.connectivity_matrices import vec_to_sym_matrix

from neuroginius.networks import group_by_networks
from neuroginius.atlas import Atlas
from neuroginius.iterables import unique

from bold_dementia.cogpred import MatrixMasker

def default_agg_func(block):
    return (block.mean(),)

class MatrixResult:
    def __init__(self, matrices, atlas) -> None:
        self.atlas = atlas
        if matrices.ndim == 2:
            matrices = matrices.reshape((1, *matrices.shape))
        self.matrices = matrices
        self._set_sorted_matrices()
    
    def _set_sorted_matrices(self):
        """Reorganize the matrices by macro labels, store
        the sorted matrices and a mapping from networks name
        to indexes in the sorted matrices
        """

        ticks, sort_index = group_by_networks(self.atlas.macro_labels)
        matrices_sort = np.ix_(sort_index, sort_index)

        self.sorted_matrices = self.matrices[:, *matrices_sort]
        new_labels = sorted(tuple(unique(self.atlas.macro_labels)))
        self.network_to_idx = pd.Series(dict(zip(
            new_labels,
            it.pairwise(ticks)
        )))

    def get_macro_matrices(self, agg_func=default_agg_func):
        """Get a matrices reorganized by networks

        Args:
            agg_func (function, optional): function to compute
            the aggregated of each cell, from the block of original
            values. Defaults to default_agg_func, which performs a simple
            mean

        Returns:
            DataFrame: summary per network of the original matrices.
        """
        for matrix in self.sorted_matrices:
            gen = self._gen_macro_values(
                matrix,
                agg_func=agg_func
            )
            comparisons = pd.DataFrame(gen, columns=["node_a", "node_b", "connectivity"])
            pivoted = comparisons.pivot(index="node_a", columns="node_b")
            yield pivoted.loc[:, "connectivity"]

    # This could be a function on its own
    def _gen_macro_values(self, sorted_matrix, agg_func):
        for network_a, network_b in it.product(self.network_to_idx.index, self.network_to_idx.index):
            loc_a, loc_b = self.network_to_idx[network_a], self.network_to_idx[network_b]
            self.block = sorted_matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]]

            yield network_a, network_b, *agg_func(self.block)
    
        
    
        

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
        axes.set_xticklabels(lbls, rotation=90)

    return axes

def region_split(label):
    return label.split("_")[-2]

class MockAtlas:
    def __init__(self, regions) -> None:
        self.macro_labels = list(map(region_split, regions))

def extract_net(mat, atlas, network):
    # Get region labels
    label_msk = np.array(atlas.macro_labels) == network
    regions = np.array(atlas.labels[label_msk])
    regions = list(np.array(regions).astype(str))

    # Extract vector and reproject to matrix
    net_masker = MatrixMasker((network,), (network,))
    if mat.ndim == 2:
        mat = mat.reshape((1, *mat.shape))
    res = net_masker.fit_transform(mat).squeeze()
    reprojected = vec_to_sym_matrix(res, diagonal=np.zeros((len(regions))))
    return reprojected, regions

def plot_network(mat, atlas, network, ax=None, **plot_kws):
    reprojected, regions = extract_net(mat, atlas, network)

    mock_atlas = MockAtlas(regions)
    
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 6))
    plot_matrix(reprojected, mock_atlas, axes=ax, **plot_kws)
    ax.set_xticklabels(np.unique(mock_atlas.macro_labels), rotation=90)
    return ax, reprojected

# Too many arguments passing between
# net_to_brain and extract_net,
# perhaps worth refactoring as object
def net_to_brain(matrix, atlas, refnet):
    reprojected, regions = extract_net(matrix, atlas, refnet)
    
    centralities = []
    for i, r in enumerate(regions):
        centralities.append(
            reprojected[i, :].sum() / 2
        )

    padded_centralities = []
    pointer = 0
    for net in atlas.macro_labels:
        if net == refnet:
            padded_centralities.append(centralities[pointer])
            pointer += 1
        else:
            padded_centralities.append(0)
    
    padded_centralities = np.array(padded_centralities)

    masker = NiftiLabelsMasker(atlas.maps)
    masker.fit()
    img = masker.inverse_transform(padded_centralities)
    return img