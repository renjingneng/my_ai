import torchinfo


def show_network(net, input_size):
    """
    show summary of network
    :param net:
    :param input_size:
    :return:
    """
    torchinfo.summary(net, input_size, verbose=2, col_names=["output_size", "num_params", "mult_adds"])
