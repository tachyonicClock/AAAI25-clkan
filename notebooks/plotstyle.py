from matplotlib import pyplot as plt
from seaborn import color_palette, set_palette

plt.rcParams.update(
    {
        # Set font family
        "font.family": "sans-serif",
        "font.sans-serif": ["Arimo"],
        # Set font size
        "font.size": 8,
        # Set maths font
        "mathtext.fontset": "custom",
        "mathtext.rm": "NewComputerModernMath",
    }
)

# Set default color palette
CMAP = color_palette("deep", as_cmap=True)

set_palette("deep")
