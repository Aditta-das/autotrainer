import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])

def kind_plot(df, label, type="bar"):
    if type == "bar":
        df[label].value_counts().plot(kind=type)
        plt.show()
    elif type == "line":
        pass

def line_plot():
    pass

def area_plot():
    pass

def bar_plot(items, values):
    plt.bar(items, values)
    plt.show()

def histogram_plot():
    # A histogram is a way of representing the frequency distribution of a variable.
    pass

def box_plot():
    pass

def pie_plot():
    pass

def scatter_plot():
    pass