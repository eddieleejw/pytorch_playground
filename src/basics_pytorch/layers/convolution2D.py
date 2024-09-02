import torch.nn as nn
import torch
import torch.nn.functional as F

class convolution2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size: tuple, stride=1, padding=0, bias=True, device = None):
        super().__init__()

        assert isinstance(kernel_size, tuple)

        self.filters = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
    

    
    def forward_slow(self, X):
        '''
        X will have shape (N, in_channels, height, width)

        this is the "slow" forward version, that iterates through the output heights and width and computes convolutiosn one at a time
        '''

        kernel_height, kernel_width = self.kernel_size

        # calculate output size
        if (((X.shape[2] - kernel_height + 2 * self.padding)/(self.stride)) % 1) == 0:
            output_height = int((X.shape[2] - kernel_height + 2 * self.padding)/(self.stride) + 1)
        else:
            raise ValueError("Convolution output height is in valid")
        
        if (((X.shape[3] - kernel_width + 2 * self.padding)/(self.stride)) % 1) == 0:
            output_width = int((X.shape[3] - kernel_width + 2 * self.padding)/(self.stride) + 1)
        else:
            raise ValueError("Convolution output width is in valid")

        # add zero padding
        padding = (self.padding, self.padding, self.padding, self.padding)
        X = F.pad(X, padding, mode = "constant", value = 0)

        output = torch.zeros(X.shape[0], self.out_channels, output_height, output_width).to(self.device)

        # we could put all the filters together into one big tensor with shape (1, out_channels, in_channels, kernel_height, kernel_width)
        # and then multiply it with a "slice" of X that has shape (n, 1, in_channels, kernel_height, kernel_width)
        # then we can multiply these point-wise to get (n, out_channels, in_channels, kernel_height, kernel_width), then sum along dim = 2, 3, 4 to get (n, out_channels, 1, 1, 1)
        # i.e. for each data point, and for each output channel, we have a single number

        for i in range(output_height):
            for j in range(output_width):
                X_slice = X[:,:, i * self.stride : i * self.stride + kernel_height, j * self.stride : j * self.stride + kernel_width] # shape (n, in_channels, kernel_height, kernel_width)
                X_slice = X_slice.unsqueeze(dim = 1) # shape (n, 1, in_channels, kernel_height, kernel_width)

                # do convolution
                # self.filters has shape (out_channels, in_channels, kernel_height, kernel_width), so it will automatically prepend a "1" for broadcasting
                conv = X_slice * self.filters # shape (n, out_channels, in_channels, kernel_height, kernel_width)

                conv = torch.sum(conv, dim = (2, 3, 4)) # shape (n, out_channel)

                # output[:, :, i, j] has shape (n, out_channel), and selects the ij-th entry for each matrix of each output channel in each data sample
                output[:, :, i, j] = conv

                # bias
                if self.bias is not None:
                    # self.bias has shape (out_channel), so adding will broadcast it by prependding a 1 to be (1, out_channel)
                    output[:, :, i, j] += self.bias

        return output

    def forward(self, X):
        '''
        This a faster forward algorithm, that circumvents iterating
        '''
        kernel_height, kernel_width = self.kernel_size

        # calculate output size
        if (((X.shape[2] - kernel_height + 2 * self.padding)/(self.stride)) % 1) == 0:
            output_height = int((X.shape[2] - kernel_height + 2 * self.padding)/(self.stride) + 1)
        else:
            raise ValueError("Convolution output height is in valid")
        
        if (((X.shape[3] - kernel_width + 2 * self.padding)/(self.stride)) % 1) == 0:
            output_width = int((X.shape[3] - kernel_width + 2 * self.padding)/(self.stride) + 1)
        else:
            raise ValueError("Convolution output width is in valid")

        # add zero padding
        padding = (self.padding, self.padding, self.padding, self.padding)
        X = F.pad(X, padding, mode = "constant", value = 0)
        self.X = X

        output = torch.zeros(X.shape[0], self.out_channels, output_height, output_width).to(self.device)

        
        '''
        F.unfold takes a in a 4D tensor, like X_unfolded, and extracts sliding blocks from it
        It will return a tensor with shape (n, in_channels * kernel_height * kernel_width, output_height * output_width)

        It's a little cleared to understand what's going on if we view it as a (n, in_channels, kernel_height, kernel_width, output_height * output_width) tensor

        Denoting this as X_unfolded,
        X_unfolded[0, 0, :, :, 0] will select one data point, just one channel, and one output, and hence will give you a (kernel_height, kernel_width) matrix which is the block of X that will be convolved to give you the selected output
        similarly, X_unfolded[0, :, :, :, 0] will give you a (in_channels, kernel_height, kernel_width) tensor, which consists of 3 (kernel_height, kernel_width) matrices stacked on each other, representing the 3 blocks of X that will be convolved for each channel respectively
        Meanwhile, X_unfolded[0, 0, 0, 0, :] will give a (output_height * output_width) vector, consisting of all the points in X that the selected data point, channel, and kernel entry (i.e. one specific entry in the kernel) will see during all the convolutions

        We want to multiply this with our filters in a batched way. The filters have shape (out_channels, in_channels, kernel_height, kernel_width)

        Since for each entry in the output, we use the same filters, we can append a "1" to the filters shape to make it (out_channels, in_channels, kernel_height, kernel_width, 1)
        
        And then we can add an extra dimension in X_unfolded to make its dimensions (n, 1, in_channels, kernel_height, kernel_width, output_height * output_width)
            This is to make it broadcastable with filters. This works because for each out_channel, we want to use the same sliding blocks, so the above will do the following:
                For each data point, copy all the blocks "out_channels" times
        
        Then, multiplying the two yields a tensor with shape (n, out_channels, in_channels, kernel_height, kernel_width, output_height * output_width)

        We want to collpase this to sum over all in_channels, kernel_heights, and kernel_widths
            This yields a tensor of shape (n, out_channels, output_height * output_width)
            We can view this as (n, out_channels, output_height, output_width), which is the desired output
        '''

        X_unfolded = F.unfold(X, kernel_size=self.kernel_size, stride = self.stride)
        X_unfolded = X_unfolded.view(X.shape[0], X.shape[1], kernel_height, kernel_width, output_height * output_width)
        X_unfolded = X_unfolded.unsqueeze(dim = 1)

        output = X_unfolded * self.filters.unsqueeze(dim = 4)

        output = torch.sum(output, dim = (2, 3, 4), keepdim=False)

        output = output.view(X.shape[0], self.out_channels, output_height, output_width)

        # TODO bias
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1).unsqueeze(2)

        return output


