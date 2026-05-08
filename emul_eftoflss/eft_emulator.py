import numpy as np 
import torch.nn as nn
import torch
import h5py as h5
import chaospy

################################################################################

# timing for debug
from contextlib import contextmanager
import time
@contextmanager
def timer(label):
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

################################################################################

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm3 = Affine()

        self.act1 = activation_fcn(in_size) 
        self.act3 = activation_fcn(in_size) 

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip           
        o3 = self.act3(self.norm3(o2))

        return o3
    
class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

"""-----------------------------------------------------------------------------
Emulator Class
-----------------------------------------------------------------------------"""

class unified_model(nn.Module):
    """
    Actual neural network that predicts loop integrals:
    Takes a list of models as its input
    These should be the dummy_model class
    """
    def __init__(self, models):
        super(unified_model, self).__init__()
        torch.set_default_dtype(torch.float32)

        emu1 = models[0] 
        emu2 = models[1] 
        emu3 = models[2] 
        emu4 = models[3]

        m1 = emu1.model
        m2 = emu2.model 
        m3 = emu3.model 
        m4 = emu4.model

        self.models = nn.ModuleList([m1, m2, m3, m4])
        
        self.register_buffer(name='samples_max', tensor=torch.Tensor(emu1.samples_max))
        self.register_buffer(name='samples_min', tensor=torch.Tensor(emu1.samples_min))

        self.register_buffer(name='pce_exponents1', tensor=torch.Tensor(emu1.exponents).long())
        self.register_buffer(name='pce_exponents2', tensor=torch.Tensor(emu2.exponents).long())
        self.register_buffer(name='pce_exponents3', tensor=torch.Tensor(emu3.exponents).long())
        self.register_buffer(name='pce_exponents4', tensor=torch.Tensor(emu4.exponents).long())

        self.register_buffer(name='pce_coefficients1', tensor=torch.Tensor(np.array(emu1.coefficients)))
        self.register_buffer(name='pce_coefficients2', tensor=torch.Tensor(np.array(emu2.coefficients)))
        self.register_buffer(name='pce_coefficients3', tensor=torch.Tensor(np.array(emu3.coefficients)))
        self.register_buffer(name='pce_coefficients4', tensor=torch.Tensor(np.array(emu4.coefficients)))

        self.register_buffer(name='pca_trfm1', tensor=torch.Tensor(emu1.pcs_nn[:-1]))
        self.register_buffer(name='pca_trfm2', tensor=torch.Tensor(emu2.pcs_nn[:-1]))
        self.register_buffer(name='pca_trfm3', tensor=torch.Tensor(emu3.pcs_nn[:-1]))
        self.register_buffer(name='pca_trfm4', tensor=torch.Tensor(emu4.pcs_nn[:-1]))

        self.register_buffer(name='pca_mean1', tensor=torch.Tensor(emu1.pcs_nn[-1]))
        self.register_buffer(name='pca_mean2', tensor=torch.Tensor(emu2.pcs_nn[-1]))
        self.register_buffer(name='pca_mean3', tensor=torch.Tensor(emu3.pcs_nn[-1]))
        self.register_buffer(name='pca_mean4', tensor=torch.Tensor(emu4.pcs_nn[-1]))

        self.register_buffer(name='y_max1', tensor=torch.tensor(emu1.y_max))
        self.register_buffer(name='y_max2', tensor=torch.tensor(emu2.y_max))
        self.register_buffer(name='y_max3', tensor=torch.tensor(emu3.y_max))
        self.register_buffer(name='y_max4', tensor=torch.tensor(emu4.y_max))

        self.register_buffer(name='y_min1', tensor=torch.tensor(emu1.y_min))
        self.register_buffer(name='y_min2', tensor=torch.tensor(emu2.y_min))
        self.register_buffer(name='y_min3', tensor=torch.tensor(emu3.y_min))
        self.register_buffer(name='y_min4', tensor=torch.tensor(emu4.y_min))

        self.register_buffer(name='pce_trfm1', tensor=torch.Tensor(emu1.pcs_pce[:-1]))
        self.register_buffer(name='pce_trfm2', tensor=torch.Tensor(emu2.pcs_pce[:-1]))
        self.register_buffer(name='pce_trfm3', tensor=torch.Tensor(emu3.pcs_pce[:-1]))
        self.register_buffer(name='pce_trfm4', tensor=torch.Tensor(emu4.pcs_pce[:-1]))

        self.register_buffer(name='pce_mean1', tensor=torch.Tensor(emu1.pcs_pce[-1]))
        self.register_buffer(name='pce_mean2', tensor=torch.Tensor(emu2.pcs_pce[-1]))
        self.register_buffer(name='pce_mean3', tensor=torch.Tensor(emu3.pcs_pce[-1]))
        self.register_buffer(name='pce_mean4', tensor=torch.Tensor(emu4.pcs_pce[-1]))
        
    def forward(self, x):
        x_norm = 2*(x - self.samples_min)/(self.samples_max - self.samples_min) - 1

        # evaluate the NN part of the models
        y1_nn = self.models[0](x_norm)
        y2_nn = self.models[1](x_norm)
        y3_nn = self.models[2](x_norm)
        y4_nn = self.models[3](x_norm)

        # inverse pca and scaling
        y1_nn = (y1_nn @ self.pca_trfm1 + self.pca_mean1)*(self.y_max1 - self.y_min1) + self.y_min1
        y2_nn = (y2_nn @ self.pca_trfm2 + self.pca_mean2)*(self.y_max2 - self.y_min2) + self.y_min2
        y3_nn = (y3_nn @ self.pca_trfm3 + self.pca_mean3)*(self.y_max3 - self.y_min3) + self.y_min3
        y4_nn = (y4_nn @ self.pca_trfm4 + self.pca_mean4)*(self.y_max4 - self.y_min4) + self.y_min4

        # evaluate the PCE part of the models
        # chaospy actually stores the coefficients to the monomials NOT the 
        # basis polynomial (legendre if X was uniform distributed, for example)
        p0 = torch.ones_like(x_norm)
        p1 = x_norm
        p2 = x_norm**2
        p3 = x_norm**3
        p4 = x_norm**4
        # shape: (batch_size, n_sparse_terms, dim=7)
        poly_stack = torch.stack([p0, p1, p2, p3, p4], dim=1).view(x.size(0), -1, 7)

        # some index setup
        batch_idx = torch.arange(x.size(0)).view(-1, 1, 1)
        dim_idx = torch.arange(7).view(1, 1, -1)
        
        # tensors containing the [phi_n1(x1), phi_n2(x2), ...] polynomials
        # shape: (batch_size, n_sparse_terms, 7)
        sparse_terms1 = poly_stack[batch_idx, self.pce_exponents1, dim_idx]
        sparse_terms2 = poly_stack[batch_idx, self.pce_exponents2, dim_idx]
        sparse_terms3 = poly_stack[batch_idx, self.pce_exponents3, dim_idx]
        sparse_terms4 = poly_stack[batch_idx, self.pce_exponents4, dim_idx]

        # multiply by the coefficients
        # the torch.prod multiplies the polynomials in [phi_n1(x1), phi_n2(x2), ...]
        # the matmul gives the coefficient for each principal component
        # resulting shape: (batch_size, n_pca_pce)
        y1_pce = torch.matmul(torch.prod(sparse_terms1, dim=-1), self.pce_coefficients1)
        y2_pce = torch.matmul(torch.prod(sparse_terms2, dim=-1), self.pce_coefficients2)
        y3_pce = torch.matmul(torch.prod(sparse_terms3, dim=-1), self.pce_coefficients3)
        y4_pce = torch.matmul(torch.prod(sparse_terms4, dim=-1), self.pce_coefficients4)

        # inverse pca transform
        y1_pce = y1_pce @ self.pce_trfm1 + self.pce_mean1
        y2_pce = y2_pce @ self.pce_trfm2 + self.pce_mean2
        y3_pce = y3_pce @ self.pce_trfm3 + self.pce_mean3
        y4_pce = y4_pce @ self.pce_trfm4 + self.pce_mean4

        # add results together
        y1 = torch.sinh(y1_nn) + torch.sinh(y1_pce)
        y2 = torch.sinh(y2_nn) + torch.sinh(y2_pce)
        y3 = torch.sinh(y3_nn) + torch.sinh(y3_pce)
        y4 = torch.sinh(y4_nn) + torch.sinh(y4_pce)

        return y1, y2, y3, y4
class dummy_model():
    """
    Dummy class for loop emulator. 
    Uses the sizes from the unified h5 file and initializes the data with zeros
    This data will be overwritten by the state_dict of the unified_model
    """
    def __init__(self, samples_shape, pca_pce_shape, pca_nn_shape, 
                 pce_coefficients_shape, pce_exponents_shape, n_layers, hidden_dim):
        
        super(dummy_model, self).__init__()
        torch.set_default_dtype(torch.float32)
        
        input_dim = int(samples_shape[0])
        output_dim = int(pca_nn_shape[0])-1
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers):
            layers.append(ResBlock(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        model = nn.Sequential(*layers)
        self.model = model
        
        self.coefficients = np.zeros(pce_coefficients_shape)
        self.exponents    = np.zeros(pce_exponents_shape)
        self.y_max = 0.0
        self.y_min = 0.0
        self.samples_max = np.zeros(samples_shape)
        self.samples_min = np.zeros(samples_shape)
        self.pcs_nn  = np.zeros(pca_nn_shape)
        self.pcs_pce = np.zeros(pca_pce_shape)


class unified_emulator():
    def __init__(self, file, device='cpu'):
        super(unified_emulator, self).__init__()
        
        # need to get the shapes of the individual emulators
        with h5.File(file+'.h5', 'r') as f:
            samples_shape = tuple(f['samples_shape'])
    
            pca_pce_shape0 = tuple(f['pca_pce_shape0'])
            pca_pce_shape1 = tuple(f['pca_pce_shape1'])
            pca_pce_shape2 = tuple(f['pca_pce_shape2'])
            pca_pce_shape3 = tuple(f['pca_pce_shape3'])

            pca_nn_shape0 = tuple(f['pca_nn_shape0'])
            pca_nn_shape1 = tuple(f['pca_nn_shape1'])
            pca_nn_shape2 = tuple(f['pca_nn_shape2'])
            pca_nn_shape3 = tuple(f['pca_nn_shape3'])

            pce_coefficients_shape0 = tuple(f['pce_coefficients_shape0'])
            pce_coefficients_shape1 = tuple(f['pce_coefficients_shape1'])
            pce_coefficients_shape2 = tuple(f['pce_coefficients_shape2'])
            pce_coefficients_shape3 = tuple(f['pce_coefficients_shape3'])

            pce_exponents_shape0 = tuple(f['pce_exponents_shape0'])
            pce_exponents_shape1 = tuple(f['pce_exponents_shape1'])
            pce_exponents_shape2 = tuple(f['pce_exponents_shape2'])
            pce_exponents_shape3 = tuple(f['pce_exponents_shape3'])

            n_layers0 = np.array(f['n_layers0'])
            n_layers1 = np.array(f['n_layers1'])
            n_layers2 = np.array(f['n_layers2'])
            n_layers3 = np.array(f['n_layers3'])

            hidden_dim0 = np.array(f['hidden_dim0'])
            hidden_dim1 = np.array(f['hidden_dim1'])
            hidden_dim2 = np.array(f['hidden_dim2'])
            hidden_dim3 = np.array(f['hidden_dim3'])
            
        model1 = dummy_model(samples_shape, pca_pce_shape0, pca_nn_shape0, \
            pce_coefficients_shape0, pce_exponents_shape0, n_layers0, hidden_dim0)
        model2 = dummy_model(samples_shape, pca_pce_shape1, pca_nn_shape1, \
            pce_coefficients_shape1, pce_exponents_shape1, n_layers1, hidden_dim1)
        model3 = dummy_model(samples_shape, pca_pce_shape2, pca_nn_shape2, \
            pce_coefficients_shape2, pce_exponents_shape2, n_layers2, hidden_dim2)
        model4 = dummy_model(samples_shape, pca_pce_shape3, pca_nn_shape3, \
            pce_coefficients_shape3, pce_exponents_shape3, n_layers3, hidden_dim3)

        self.model = unified_model([model1, model2, model3, model4])
        state_dict = torch.load(file+'.pt', map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.cmodel = torch.compile(self.model, fullgraph=True, mode="reduce-overhead")
        dummyx = torch.zeros(samples_shape)
        for i in range(10):
            self.predict(dummyx)
        print('[eft] loaded model:', file)

    def predict(self, x):
        ### NN part
        x = np.atleast_2d(x)
        with torch.inference_mode():
            y0, y1, y2, yj = self.cmodel(torch.as_tensor(x,dtype=torch.float32))
        return y0, y1, y2, yj
