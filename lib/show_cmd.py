import torch


def show_network_output(net, network_type='cnn'):
    print('|----------[show_network_output-START]----------|')
    if network_type == 'cnn':
        X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
    else:
        print('error:unknown type!')
    print('|----------[show_network_output-END]------------|')
