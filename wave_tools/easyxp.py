
"""
@email : xianpuji@hhu.edu.cn
"""

"""
easyxp - A lightweight Python library for creating quiver plot legends
"""


import matplotlib.pyplot as plt
import matplotlib
from typing import Literal
from matplotlib.patches import Rectangle

LocationOption = Literal['lower right', 'lower left', 'upper right', 'upper left']

def simple_quiver_legend(
    ax: plt.Axes,
    quiver: matplotlib.quiver.Quiver,
    reference_value: float = 10.0,
    unit: str = 'm/s',
    legend_location: LocationOption = 'lower right',
    box_width: float = 0.11,
    box_height: float = 0.15,
    text_offset: float = 0.02,
    font_size: int = 7,
    label_separation: float = 0.1,
    box_facecolor: str = 'white',
    box_edgecolor: str = 'k',
    box_linewidth: float = 0.8,
    zorder:float=10
) -> None:
    """
    Create a compact legend for quiver plots in specified axes corner
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object
    quiver : matplotlib.quiver.Quiver
        Quiver plot object
    reference_value : float, optional
        Reference value for legend arrow (default: 10.0)
    unit : str, optional
        Measurement unit (default: 'm/s')
    legend_location : {'lower right', 'lower left', 'upper right', 'upper left'}, optional
        Legend position (default: 'lower right')
    box_width : float, optional
        Legend box width in axes coordinates (default: 0.15)
    box_height : float, optional
        Legend box height in axes coordinates (default: 0.12)
    text_offset : float, optional
        Vertical text offset from arrow (default: 0.015)
    font_size : int, optional
        Font size for legend text (default: 7)
    label_separation : float, optional
        Distance between arrow and text (default: 0.06)
    box_facecolor : str, optional
        Box background color (default: 'white')
    box_edgecolor : str, optional
        Box border color (default: 'k')
    box_linewidth : float, optional
        Box border line width (default: 0.8)
    """

    # Determine center coordinates based on location
    position_map = {
        'lower right': (1 - box_width/2, box_height/2),
        'lower left': (box_width/2, box_height/2),
        'upper right': (1 - box_width/2, 1 - box_height/2),
        'upper left': (box_width/2, 1 - box_height/2)
    }

    try:
        center_x, center_y = position_map[legend_location]
    except KeyError:
        raise ValueError(
            f"Invalid location: {legend_location}. "
            "Valid options are: 'lower right', 'lower left', 'upper right', 'upper left'"
        )

    # Create background box
    rect = Rectangle(
        (center_x - box_width/2, center_y - box_height/2),
        box_width, box_height,
        transform=ax.transAxes,
        facecolor=box_facecolor,
        edgecolor=box_edgecolor,
        linewidth=box_linewidth
    )
    ax.add_patch(rect)

    # Create label text
    label_text = f'{reference_value} {unit}' if unit else str(reference_value)

    # Add quiver key
    ax.quiverkey(
        quiver,
        X=center_x,
        Y=center_y + text_offset,
        U=reference_value,
        label=label_text,
        labelpos='S',
        coordinates='axes',
        fontproperties={
            'size': font_size,
            'weight': 'normal'
        },
        labelsep=label_separation,
        zorder=zorder
    )
    
    