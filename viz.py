import numpy as np
import matplotlib.pyplot as plt
try:
    from colorama import Fore
    colorama_available = True
except ImportError:
    colorama_available = False


def render_pretty(text):
    print(text)


def render_color(text):
    if colorama_available:
        return f'{Fore.GREEN}{text}{Fore.RESET}'
    return text


def q_heatmap(data):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()