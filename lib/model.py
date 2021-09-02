import torch
import visdom

import utility


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())



def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def train_1(net, train_iter, test_iter, num_epochs, lr):
    """
    Layer: Linear && Conv2d
    Optim: SGD
    Loss: CrossEntropyLoss
    Purpose: img classification
    """
    # step1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.apply(init_weights)
    train_l = 0
    train_acc = 0
    test_acc = 0
    metric = utility.Accumulator(3)  # Sum of training loss, sum of training accuracy, no. of examples
    metric_eval = utility.Accumulator(2)  # No. of correct predictions, no. of predictions
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    timer, num_batches = utility.Timer(), len(train_iter)
    vis = visdom.Visdom()
    vis.close(win="lineset")
    vis.line(X=[0], Y=[train_l], win="lineset", name="train_l", update='append')
    vis.line(X=[0], Y=[train_acc], win="lineset", name="train_acc", update='append')
    vis.line(X=[0], Y=[test_acc], win="lineset", name="test_acc", update='append')
    # step2
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            # step2.1
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            timer.stop()
            # step2.2
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            vis.line(X=[epoch + (i + 1) / num_batches], Y=[train_l], win="lineset", name="train_l", update='append')
            vis.line(X=[epoch + (i + 1) / num_batches], Y=[train_acc], win="lineset", name="train_acc", update='append')
        # step2.3
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                y = y.to(device)
                metric_eval.add(accuracy(net(X), y), y.numel())
        test_acc = metric_eval[0] / metric_eval[1]
        vis.line(X=[epoch + 1], Y=[test_acc], win="lineset", name="test_acc", update='append')
