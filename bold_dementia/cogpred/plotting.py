import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from bold_dementia.cogpred import plot_matrix


def plot_haufe_pattern(maps, disp_atlas, cats, cmap="seismic"):
    bound = np.max(np.abs(maps))
    bounds=(-bound, bound)

    fig = plt.figure(figsize=(9, 12))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(cats), 1),
                    axes_pad=0.4,
                    share_all=False,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="2%",
                    cbar_pad=0.1
                    )

    for i, ax in enumerate(grid):
        val = maps[i]
        _ = plot_matrix(
            val,
            disp_atlas,
            axes=ax,
            bounds=bounds,
            cmap="seismic",
            cbar=False
        )
        ax.set_title(cats[i])
        # Keep ticks only where we have labels
        if i <= 1:
            ax.xaxis.set_ticks_position('none') 
            
    norm = mpl.colors.Normalize(vmin=-bound, vmax=bound, clip=False)
    grid.cbar_axes[0].colorbar(ScalarMappable(norm=norm, cmap=cmap))
    
    cax = grid.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text("Haufe's pattern value, ambient space")

    return fig