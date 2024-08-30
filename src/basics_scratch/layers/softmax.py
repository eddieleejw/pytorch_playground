# TODO
import torch
from module.module import module

class softmax(module):
    '''
    applies softmax to incoming data

    since data is given as rows of features, the softmax is done along the row
    '''

    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        X has shape n x input_dim

        output should also have shape n x input_dim
        '''
        exped = torch.exp(X)
        out = exped / torch.sum( exped, dim = 1, keepdim=True) # for broadcasting reasons, we want to keep the sum having shape (n,1), so it can broadcase with exped which has shape (n, input_dim)
        self.out = out

        return out

    def backward(self, grad_output):
        # TODO
        '''
        the final output dL/dX is going to have shape n x input_dim, because that's the shape that X has

        but the calculation to get there is a little difficult

        this is because we need to compute the Jacobian matrix for EACH ROW in self.out

        let f(x) := softmax(x), where x \in R^d (d := input_dim)
            note: f: R^d -> R^d

        then for one row x of X, dL/dx = dL/df * df/dx
            dL/dx \in R^d
            dL/df \in R^d (i.e. one row of grad_output)
            df/dx \in R^(d x d)

        so we need to calculate the Jacobian df/dx

        for convenience, denote J := df/dx, and J_ij the i-th row and j-th column entry of J

        then J_ij = d f_i / d x_j

        Recall that f_i = e^{x_i} / \sum_k e^{x_k}

        when i != j: - f_i x f_j (by chain rule) 
        when i == j: f_i x (1 - f_i) (by product and chain rule)

        Hence: (i != j)
            J_ij = - f_i f_j
            J_ii = f_i (1 - f_i) = f_i - f_i f_i
        
        Note that the matrix -f(x)f(x)^T has ij-th entry -f_i f_j, which is very similar to the Jacobian above
            We only need to account for an additional f_i term in all diagonals. So we can succintly write
            J = diag(f(x)) - f(x)f(x)^T
                where diag(f(x)) is a (d x d) diagonal matrix, where the ii-th entry is f(x)_i
        
        Great! Now we need to multiply the corresponding row of grad_output (i.e. dL/df) with J (i.e. df/dx)
        But what's the correct way?
            Well in this case, J is a symmetric matrix, so it doesn't matter. We can just line up the dimensions, since we know we expect dL/dx to be in R^d
            so J * dL/df is correct
        
        But as a note, what if J was not symmetric? What do we do
            For convenience, denote a := dL/dx, b := dL/df, J := df/dx

            We know by definition that a_i = dL/dx_i
            and then by chain rule, we can write out a_i = \sum_k dL/df_k * df_k/dx_i
                Note that we get this sum over f_k, because L depends on f_1, ..., f_n, and each of those in turn depends on x_i
            
            We know that b = [dL/df_1  ...  dL/df_n]^T
            and J_ij = df_i/dx_i

            From this, we can piece together that J^T b gives a vector, such that the i-th entry has form \sum_k dL/df_k * df_k/dx_i
                Which is exactly a_i

            Hence a = J^T b
            
            Again, in our case, J is symmetric i.e. J^T = T
            So a = Jb, as above
        
        I implement a row-wise to showcase the math more clearly,
        but also give a batched implementation to allow parallelisation
            The batched implementation is more than 100x faster on my machine
        '''
        PARALLEL = True
        if not PARALLEL:
            # example of how to do the calculaton row by row

            # initialise an empty (d x d) output matrix
            grad_input = torch.zeros_like(grad_output)

            # calculate each row of grad_input separately

            d = grad_output.shape[0]

            for i in range(d):
                f = self.out[i].unsqueeze(dim=1) # f(x)
                g = grad_output[i].unsqueeze(dim=1)

                assert f.shape == torch.Size([5,1])
                assert f.shape == g.shape
            
                # calculate jacobian
                J = torch.diag(f.squeeze()) - torch.matmul(f, f.T)

                # calculate dL/dx
                dx = torch.matmul(J, g)

                # assign to grad_input
                grad_input[i] = dx.squeeze()
        else:
            # doing the same thing but parallelising as batch operations

            S = self.out # shape = (n, d)

            S = S.unsqueeze(2) # shape = (n, d, 1)

            S_T = S.transpose(1, 2) # shape = (n, 1, d)

            # the following matmul will iterate through each batch (i.e. i in [n]), and do the multiplication S[i] x S_T[i], resulting in a d x d matrix for each batch, and then stacks them to make a tensor of shape (n, d, d) (i.e. n matrices of shape (d x d))
            outer_product = torch.matmul(S, S_T) # shape = (n, d, d), where outer_product[i] is a matrix defined as S[i] x S[i]^T, where S[i] := f^(i) = f(x^(i)), i.e. the output vector correspoding to the ith input vector

            # create the diagonal matrix
            # self.out has shape (n, d). diag_embed will take each self.out[i] and turn them into (d, d) diagonal matrices where the diagonsl are self.out[i], and then stacks "n" of them to output shape (n, d, d) i.e. "n" diagonal matrices of shape (d, d)
            diag_matrix = torch.diag_embed(self.out)

            # do elementwise subtraction to get the batched Jacobian of size (n, d, d), where J[i] is the Jacobian of the i-th sample
            J = diag_matrix - outer_product

            # calculate dL/dx
            # grad_output has shape (n, d). to make it broadcastable, we want to make it shape (n, d, 1) so that torch.matmul will calculate J[i] x dx[i]
            dx = torch.matmul(J, grad_output.unsqueeze(dim = 2))

            # dx has shape (n, d, 1), we want to turn it back to (n, d)
            grad_input = dx.squeeze(dim = 2)
        
        return grad_input




