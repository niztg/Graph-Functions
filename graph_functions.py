import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
from math import ceil

__all__ = (
    'prepare_bar',
    'prepare_plot',
    'prepare_scatterplot',
    'prepare_linegraph',
    'prepare_manylines',
    'prepare_hbar',
    'prepare_2bar'
)


def prepare_plot(
        plt_values: dict,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        color: str = 'white',
        grid: bool = True,
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)

    ax.set_facecolor(background_color)

    master_y_values = []
    for x in plt_values.values():
        master_y_values.append(x[1])

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(master_y_values), kwargs.get('max') or max(master_y_values))

    if ((mini := kwargs.get('min')) and mini < 0) or (min(master_y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    for x in plt_values.values():
        plt.scatter(x[0], x[1], color=color)

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    if kwargs.get('annotate'):
        for x, y in plt_values.items():
            plt.annotate(f"{x}\n", (y[0], y[1]), fontsize='x-small', color=prefered_def_colors,
                         horizontalalignment='center', verticalalignment='bottom')

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    if kwargs.get('br_pred'):
        rg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        plt.plot(rg, rg, linestyle='dotted', linewidth=0.8)

    return plt.show()


def prepare_bar(
        x_values: List[str],
        y_values: List[Union[float, int]],
        colors: Union[List[str], str],
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)

    ax.set_facecolor(background_color)

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('alt_colors'):
        if len(colors) != 2:
            print("Nah")
            return

        colors *= int(ceil(len(y_values) / 2))

    if max := kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(y_values), kwargs.get('max') or max(y_values))

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    else:
        plt.xticks(rotation=kwargs.get('rotation') or 0)

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    plt.bar(x=x_values, height=y_values, color=colors, width=0.6)

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if not (add_text := kwargs.get('add_text')):
        for index, value in enumerate(y_values):

            if not kwargs.get('dont_add_names'):
                text_to_add = f"{x_values[index]}\n"

            else:
                text_to_add = ""

            if kwargs.get('add_signs'):
                score = lambda i: ("+" if i > 0 else "") + str(i)
                text_to_add += score(value)

            else:
                text_to_add += str(value)

            plt.text(index, value, text_to_add, horizontalalignment='center',
                     verticalalignment='bottom', fontsize='xx-small', color=prefered_def_colors)

    elif add_text == 'no':
        pass

    elif add_text:
        for index, value in enumerate(y_values):
            text_to_add = f"{x_values[index]}\n{value}"

            plt.text(index, value, text_to_add, horizontalalignment='center',
                     verticalalignment='bottom', fontsize='xx-small', color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    if ((mini := kwargs.get('min')) and mini < 0) or (min(y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    return plt.show()


def prepare_scatterplot(
        plt_values: dict,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        legend_loc: str = 'best',
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)
    y_values = list(plt_values.values())
    amt = len(plt_values.keys())

    colors = kwargs.get('color')

    if colors:
        if len(colors) < amt:
            raise ValueError("Not enough colors imo")

    if kwargs.get('lines'):
        for x in plt_values.values():
            if not colors:
                plt.plot(x[0], x[1], marker='o', linestyle='dotted')
            else:
                plt.plot(x[0], x[1], marker='o', linestyle='dotted', color=colors[list(plt_values.values()).index(x)])

            if kwargs.get('annotate'):
                label_s = kwargs.get('labels')
                if not label_s:
                    raise ValueError('Need labels tubby')

                else:
                    current_label = label_s[y_values.index(x)]
                    for i, label in zip(zip(x[0], x[1]), current_label):
                        plt.annotate(f"{label}\n", (i[0], i[1]), fontsize='xx-small', color=prefered_def_colors,
                                     horizontalalignment='center', verticalalignment='bottom')

    else:
        for x in y_values:
            if colors:
                ax.scatter(x[0], x[1], color=colors[y_values.index(x)])

            else:
                ax.scatter(x[0], x[1])

            if kwargs.get('annotate'):
                label_s = kwargs.get('labels')
                if not label_s:
                    raise ValueError('Need labels tubby')

                else:
                    current_label = label_s[y_values.index(x)]
                    for i, label in zip(zip(x[0], x[1]), current_label):
                        plt.annotate(f"{label}\n", (i[0], i[1]), fontsize='xx-small', color=prefered_def_colors,
                                     horizontalalignment='center', verticalalignment='bottom')

    if kwargs.get('legend'):
        plt.legend(list(plt_values.keys()), loc=legend_loc, framealpha=1.0, edgecolor='w')

    master_y_values = []
    for x in plt_values.values():
        master_y_values += x[1]

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(master_y_values), kwargs.get('max') or max(master_y_values))

    if ((mini := kwargs.get('min')) and mini < 0) or (min(master_y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    ax.set_facecolor(background_color)

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    return plt.show()


def prepare_linegraph(
        x_values: List[str],
        y_values: List[Union[float, int]],
        line_color: str = 'white',
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        marker: bool = False,
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param grid:
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)

    ax.set_facecolor(background_color)

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(y_values), kwargs.get('max') or max(y_values))

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    else:
        plt.xticks(rotation=kwargs.get('rotation') or 0)

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    if marker:
        plt.plot(x_values, y_values, color=line_color, marker=kwargs.get('mark') or 'o')

    if kwargs.get('annotate'):
        for x, y in zip(x_values, y_values):
            plt.annotate(f"{str(y)}\n", (x, y), fontsize='x-small', color=prefered_def_colors,
                         horizontalalignment='center', verticalalignment='bottom')

    else:
        plt.plot(x_values, y_values, color=line_color)

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    if ((mini := kwargs.get('min')) and mini < 0) or (min(y_values) < 0):
        plt.axhline(0, color=prefered_def_colors, linewidth=0.5)

    if (line := kwargs.get('line')):
        plt.axhline(line, color=kwargs.get('lcolor') or prefered_def_colors, linewidth=0.8)

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    return plt.show()


def prepare_manylines(
        plt_values: dict,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)
    y_values = list(plt_values.values())

    _dicts = []

    amt = len(y_values[0])
    colors = kwargs.get('color')

    if colors:
        if len(colors) != amt:
            raise ValueError("Not enough colors imo")

    for x in range(amt):
        new_dict = {}
        for i, k in plt_values.items():
            new_dict[i] = k[x]
        _dicts.append(new_dict)

    for dataset in _dicts:
        if not colors:
            plt.plot(list(dataset.keys()), list(dataset.values()))
        else:
            plt.plot(list(dataset.keys()), list(dataset.values()), color=colors[_dicts.index(dataset)])

    if kwargs.get('annotate'):
        for i in _dicts:
            for x, y in i.items():
                plt.annotate(f"{str(y)}\n", (x, y), fontsize='x-small', color=prefered_def_colors,
                             horizontalalignment='center', verticalalignment='bottom')

    if labels := kwargs.get('legend'):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        legend = ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
        for text in legend.get_texts():
            text.set_color(prefered_def_colors)

    master_y_values = []
    for x in y_values:
        master_y_values.append(x[1])

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(master_y_values), kwargs.get('max') or max(master_y_values))

    if ((mini := kwargs.get('min')) and mini < 0) or (min(master_y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    ax.set_facecolor(background_color)

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    if kwargs.get('fill_between'):
        if len(list(plt_values.values())[0]) != 2:
            print("Nah")
            return

        fvs = [v[0] for v in plt_values.values()]
        svs = [v[1] for v in plt_values.values()]
        ax.fill_between(list(plt_values.keys()), fvs, svs, where=(np.array(fvs) > np.array(svs)), color='g', alpha=0.3,
                        interpolate=True)
        ax.fill_between(list(plt_values.keys()), fvs, svs, where=(np.array(svs) > np.array(fvs)), color='r', alpha=0.3,
                        interpolate=True)

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    return plt.show()


def prepare_hbar(
        x_values: List[str],
        y_values: List[Union[float, int]],
        colors: Union[List[str], str],
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        **kwargs
):
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)

    ax.set_facecolor(background_color)

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if max := kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(y_values), kwargs.get('max') or max(y_values))

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    else:
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    plt.barh(x_values, y_values, color=colors)

    plt.xlabel(x_label, color=prefered_def_colors)
    plt.ylabel(y_label, color=prefered_def_colors)

    if not (add_text := kwargs.get('add_text')):
        for index, value in enumerate(y_values):

            if not kwargs.get('dont_add_names'):
                text_to_add = f" {value}"

            else:
                text_to_add = ""

            if kwargs.get('add_signs'):
                score = lambda i: ("+" if i > 0 else "") + str(i)
                text_to_add += score(value)

            plt.annotate(text_to_add, (value, index), color=prefered_def_colors)

    elif add_text == 'no':
        pass

    elif add_text:
        for index, value in enumerate(y_values):
            text_to_add = f"{x_values[index]}\n{value}"

            plt.annotate(text_to_add, (value, index), color=prefered_def_colors, fontsize='small')

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['left'].set_color(prefered_def_colors)
        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    if ((mini := kwargs.get('min')) and mini < 0) or (min(y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1)
    ax.set_axisbelow(True)

    return plt.show()


def prepare_2bar(
        x_values: List[str],
        fsty_values: List[Union[float, int]],
        sndy_values: List[Union[float, int]],
        labels: List[str],
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        three_bar: bool = False,
        try_values: List[Union[float, int]] = None,
        prefered_def_colors: str = 'white',
        background_color: str = '#242629',
        grid: bool = True,
        **kwargs
):
    """
    This function prepares a stat-graph automatically
    :param x_values: x-values
    :param y_values: y-values
    :param colors:   colors each bar is, in order
    :param x_label:  label of the x-axis
    :param y_label:  label of the y-axis
    :param title:    title of the graph
    :param prefered_def_colors: The prefered default color, this is automatically white
    :param background_color:    The background color
    :return:
    """
    fig = plt.figure(facecolor=background_color, figsize=(14, 7.5))
    ax = fig.add_subplot(111)

    x_axis = np.arange(len(x_values))

    if not three_bar:
        rect1 = ax.bar(x_axis - 0.2, fsty_values, 0.4, color=kwargs.get('color_1') or '#42c01e', label=labels[0])
        rect2 = ax.bar(x_axis + 0.2, sndy_values, 0.4, color=kwargs.get('color_2') or '#ee210e', label=labels[1])
        rects = [rect1, rect2]

    else:
        rect1 = ax.bar(x_axis - 0.2, fsty_values, 0.2, color=kwargs.get('color_1') or '#42c01e', label=labels[0])
        rect2 = ax.bar(x_axis, sndy_values, 0.2, color=kwargs.get('color_2') or '#ee210e', label=labels[1])
        rect3 = ax.bar(x_axis + 0.2, try_values, 0.2, color=kwargs.get('color_3') or '#75a3e0', label=labels[2])
        rects = [rect1, rect2, rect3]

    master_y_values = fsty_values + sndy_values

    min_func = lambda __min: min(__min) if min(__min) < 0 else 0

    if kwargs.get('max') or kwargs.get('min'):
        ax.set_ylim(kwargs.get('min') or min_func(master_y_values), kwargs.get('max') or max(master_y_values))

    if ((mini := kwargs.get('min')) and mini < 0) or (min(master_y_values) < 0):
        plt.axhline(0, color='white', linewidth=0.5)

    if kwargs.get('annotate'):
        for x in rects:
            for rect in x:
                height = rect.get_height()

                if kwargs.get('add_signs'):
                    score = lambda i: ("+" if i > 0 else "") + str(i)

                    ax.annotate('{}'.format(score(height)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', color=prefered_def_colors, fontsize='small')

                else:
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', color=prefered_def_colors, fontsize='small')

    ax.set_facecolor(background_color)

    plt.xticks(x_axis, x_values)

    if not kwargs.get('y_ticks'):
        plt.yticks([])

    if not kwargs.get('x_ticks'):
        plt.xticks([])

    else:
        plt.xticks(rotation=kwargs.get('rotation') or 0)
        for axis in ['x', 'y']:
            if axis == 'x':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('x_size') or 'xx-small')

            if axis == 'y':
                ax.tick_params(axis, colors=prefered_def_colors, labelsize=kwargs.get('y_size') or 'xx-small')

    if kwargs.get('legend'):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
        for text in legend.get_texts():
            text.set_color(prefered_def_colors)

    if x_label:
        plt.xlabel(x_label, color=prefered_def_colors)

    if y_label:
        plt.ylabel(y_label, color=prefered_def_colors)

    if title:
        plt.title(title, color=prefered_def_colors)

    if description := kwargs.get('description'):
        plt.figtext(0.5, 0.01, description, color=prefered_def_colors)

    if kwargs.get('del_spines'):
        for spine in ax.spines.values():
            spine.set_edgecolor(background_color)

        ax.spines['bottom'].set_color(prefered_def_colors)
    else:
        [spine.set_edgecolor(prefered_def_colors) for spine in ax.spines.values()]

    ax.grid(grid, linewidth=kwargs.get('axis_linewidth') or 0.1, linestyle=kwargs.get('axis_linestyle') or 'solid')

    ax.set_axisbelow(True)

    return plt.show()
