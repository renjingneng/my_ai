import torch
import torch.nn as nn

# torch.view vs torch.reshape
def test1():
    """
    1.torch.view merely creates a view of the original tensor. The new tensor will always share its data with the
    original tensor.
    2.torch.reshape doesn't impose any contiguity constraints, but also doesn't guarantee data sharing.
    """
    print(torch.arange(10).is_contiguous())
    print(torch.arange(10).view(2, -1))
    print(torch.arange(10).reshape(2, -1))


# torch.cat vs torch.split /torch.chunk
def test2():
    # torch.cat
    temp1 = torch.arange(10).view(2, 5)
    temp2 = torch.arange(10).view(2, 5)
    print(torch.cat([temp1, temp2], 0).shape)
    print(torch.cat([temp1, temp2], 1).shape)
    # torch.split
    temp = torch.arange(20).view(4, 5)
    print(torch.split(temp, 2))
    print(torch.split(temp, [2, 3], 1))
    # torch.chunk, len_chunk = ceiling(total_len/num_chunks) ,len_last_chunk <= len_chunk
    temp = torch.arange(65).view(13, 5)
    print([len(item) for item in torch.chunk(temp, 3)])


# In-place operations
def test3():
    """
    In-place operations save some memory, but can be problematic when computing derivatives because of an immediate
    loss of history. Hence, their use is discouraged.
    """
    tensor1 = torch.arange(20).view(4, 5)
    tensor2 = torch.arange(2, 22).view(4, 5)
    tensor1.add_(1)
    print(tensor1)
    tensor1.copy_(tensor2)
    print(tensor1)
    tensor1.t_()
    print(tensor1.shape)


# detach().clone()  vs   clone().detach()
def test4():
    A = torch.rand(2, 2)
    A.detach().clone()  # slightly more efficient
    A.clone().detach()


# max、index_select 、masked_select、gather、nonzero
def test5():
    # max
    A = torch.rand(2, 3,6,9)
    print(A.max(0)[0].shape)
    print(A.max(0)[1].shape)
    # gather : Gathers values along an axis specified by dim.
    t = torch.tensor(
        [[1, 2]
            , [3, 4]])
    result = torch.gather(t, 0, torch.tensor(
        [[0, 0]
            , [1, 0]]))
    print(result)


def rnn_test1():
    rnn = nn.LSTM(4, 5, num_layers=1,bidirectional=True)
    x = torch.randn(1, 1, 4)
    h0 = torch.randn(2, 1, 5)
    c0 = torch.randn(2, 1, 5)
    y_hat, (hn, cn) = rnn(x, (h0, c0))
    print(y_hat.shape)
    print(hn.shape)
    print(cn.shape)

def rnn_test2():
    rnn = nn.LSTM(4, 5, num_layers=1,bidirectional=False)
    x = torch.randn(6, 3, 4)
    h0 = torch.randn(1, 3, 5)
    c0 = torch.randn(1, 3, 5)
    y_hat, (hn, cn) = rnn(x, (h0, c0))
    print(y_hat.shape)
    print(hn.shape)
    print(cn.shape)


if __name__ == '__main__':
    test5()
