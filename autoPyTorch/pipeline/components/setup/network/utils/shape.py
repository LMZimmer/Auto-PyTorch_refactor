from typing import List
import warnings


def get_shaped_neuron_counts(
        shape: str,
        in_feat: int,
        out_feat: int,
        max_neurons: int,
        layer_count: int) -> List[int]:
    counts = []  # type: List[int]

    if layer_count <= 0:
        return counts

    if layer_count == 1:
        counts.append(out_feat)
        return counts

    max_neurons = max(in_feat, max_neurons)
    # https://mikkokotila.github.io/slate/#shapes

    if shape == 'brick':
        #
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |___  ___|
        #
        for _ in range(layer_count - 1):
            counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'triangle':
        #
        #        /  \
        #       /    \
        #      /      \
        #     /        \
        #    /          \
        #   /_____  _____\
        #
        previous = in_feat
        step_size = int((max_neurons - previous) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous + step_size
            counts.append(previous)
        counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'funnel':
        #
        #   \            /
        #    \          /
        #     \        /
        #      \      /
        #       \    /
        #        \  /
        #
        previous = max_neurons
        counts.append(previous)

        step_size = int((previous - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous - step_size
            counts.append(previous)

        counts.append(out_feat)

    if shape == 'long_funnel':
        #
        #   |        |
        #   |        |
        #   |        |
        #    \      /
        #     \    /
        #      \  /
        #
        brick_layer = int(layer_count / 2)
        funnel_layer = layer_count - brick_layer
        counts.extend(get_shaped_neuron_counts(
            'brick', in_feat, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
            'funnel', in_feat, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: long funnel layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'diamond':
        #
        #     /  \
        #    /    \
        #   /      \
        #   \      /
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 2) + 1
        funnel_layer = layer_count - triangle_layer
        counts.extend(get_shaped_neuron_counts(
            'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        remove_triangle_layer = len(counts) > 1
        if (remove_triangle_layer):
            # remove the last two layers since max_neurons == out_features
            # (-> two layers with the same size)
            counts = counts[0:-2]
        counts.extend(get_shaped_neuron_counts(
            'funnel',
            max_neurons,
            out_feat,
            max_neurons,
            funnel_layer + (2 if remove_triangle_layer else 0)))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: diamond layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'hexagon':
        #
        #     /  \
        #    /    \
        #   |      |
        #   |      |
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 3) + 1
        funnel_layer = triangle_layer
        brick_layer = layer_count - triangle_layer - funnel_layer
        counts.extend(get_shaped_neuron_counts(
            'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        counts.extend(get_shaped_neuron_counts(
            'brick', max_neurons, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
            'funnel', max_neurons, out_feat, max_neurons, funnel_layer))

        if len(counts) != layer_count:
            warnings.warn("\nWarning: hexagon layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'stairs':
        #
        #   |          |
        #   |_        _|
        #     |      |
        #     |_    _|
        #       |  |
        #       |  |
        #
        previous = max_neurons
        counts.append(previous)

        if layer_count % 2 == 1:
            counts.append(previous)

        step_size = 2 * int((max_neurons - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(int(layer_count / 2 - 1)):
            previous = previous - step_size
            counts.append(previous)
            counts.append(previous)

        counts.append(out_feat)

        if len(counts) != layer_count:
            warnings.warn("\nWarning: stairs layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    return counts
