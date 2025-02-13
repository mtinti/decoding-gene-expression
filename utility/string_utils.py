import matplotlib.pyplot as plt
import seaborn as sns


def multiple_replace(string):
    if string == 'TE':
        return string
    string=string.replace('count_tracts','ct')
    string=string.replace('count','c')
    string=string.replace('mismatch','m')
    string=string.replace('utr_3','3utr')   
    string=string.replace('utr_5','5utr')  
    string=string.replace('optimal','opt')
    string=string.replace('m_2','m2')
    string=string.replace('m_0','m0')
    string=string.replace('m_0','m0')
    string=string.replace('T','U')
    return string


pattern_colors = {
    'utr': sns.color_palette("tab10")[1],  # Using seaborn color palette
}

def color_yticks_by_patterns(ax, pattern_colors, default_color='black'):
    """
    Colors the ytick labels of a matplotlib axis based on multiple string patterns.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis object to modify
    pattern_colors : dict
        Dictionary where keys are string patterns to match at start
        and values are seaborn color names or hex colors
        e.g., {'utr': 'coral', 'gene': 'steelblue', 'mir': 'seagreen'}
    default_color : str, optional
        Color to use when no pattern matches (default 'gray')
    """
    # Get the ytick labels
    yticks = ax.get_yticklabels()
    
    # Set colors based on patterns
    for label in yticks:
        text = label.get_text()
        # Default color if no pattern matches
        color = default_color
        
        # Check each pattern
        for pattern, col in pattern_colors.items():
            if pattern in text:#.startswith(pattern):
                color = col
                break
                
        label.set_color(color)
    
    # Force the plot to update
    ax.figure.canvas.draw_idle()

#tab10
#husl