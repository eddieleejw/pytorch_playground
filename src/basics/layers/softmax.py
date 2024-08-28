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
        '''
        
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
        
        return grad_input




