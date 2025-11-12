import sys
import os
import random
# from torch_geometric.nn import knn
import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from .utils import toNP
from .geometry import to_basis, from_basis
import math
import trimesh
from pyFM.mesh import TriMesh
from torch_geometric.data import Batch
from .data import DiffusionData
from .transforms import DiffusionOperatorsTransform





class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if(self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class Meyer(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf=Nf

        if scales is None:
            scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        self.g = [lambda x: kernel(scales[0] * x, 'scaling_function')]

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        def kernel(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asarray(x)

            l1 = 2/3.
            l2 = 4/3.  # 2*l1
            l3 = 8/3.  # 4*l1

            def v(x):
                return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == 'scaling_function':
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kernel_type == 'wavelet':
                r[r2ind] = np.sin((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((np.pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise ValueError('Unknown kernel type {}'.format(kernel_type))

            return r


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output: 
        #   gs: [Nf,K,], pytorch tensor
        evals=evals.numpy()
        gs=np.expand_dims(self.g[0](evals),0)

        for s in range(1, self.Nf):
            gs=np.concatenate((gs,np.expand_dims(self.g[s](evals),0)),0)
        
        return torch.from_numpy(gs.astype(np.float32))

from .utils import dist_mat, nn_search

class RFMNet(torch.nn.Module):
    def __init__(self,C_in, C_out, is_mwp=True):
        
        super().__init__()
        self.is_mwp=is_mwp
        self.feat_extrac=DiffusionNet(C_in=C_in, C_out=C_out)
        self.criterion=torch.nn.MSELoss()
     
    
    def forward(self,descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(descs_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(descs_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)

        elevals_x=elevals_x[0][:100]
        elevals_y=elevals_y[0][:100]  
        elevecs_x=elevecs_x[:,:100]
        elevecs_y=elevecs_y[:,:100]  

        elevecs_x=elevecs_x.to(dtype=torch.float32)
        elevecs_y=elevecs_y.to(dtype=torch.float32)  

        BT_A_B = elevecs_x.T @ torch.diag(massvec_x) @ elevecs_x
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_x = elevecs_x @ BT_A_B_inv_sqrt

        BT_A_B = elevecs_y.T @ torch.diag(massvec_y) @ elevecs_y
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_y = elevecs_y @ BT_A_B_inv_sqrt


      
        elgs_x=[]
        positive_evals_x = elevals_x[elevals_x > 0].cpu()
        negative_evals_x = elevals_x[elevals_x < 0].cpu()
        for i in range(len(elevals_x)):
            if elevals_x[i]>0:
                elgs_x.append(Meyer(positive_evals_x[-1],Nf=6)(elevals_x[i].cpu()))
            else:
                elgs_x.append(Meyer(negative_evals_x[-1],Nf=6)(elevals_x[i].cpu())) 
        elgs_x=torch.stack(elgs_x)
        elgs_x=elgs_x.T
        elgs_x=elgs_x.to("cuda:0")

        elgs_y=[]
        positive_evals_y = elevals_y[elevals_y > 0].cpu()
        negative_evals_y = elevals_y[elevals_y < 0].cpu()
        for i in range(len(elevals_y)):
            if elevals_y[i]>0:
                elgs_y.append(Meyer(positive_evals_y[-1],Nf=6)(elevals_y[i].cpu()))
            else:
                elgs_y.append(Meyer(negative_evals_y[-1],Nf=6)(elevals_y[i].cpu())) 
        elgs_y=torch.stack(elgs_y)
        elgs_y=elgs_y.T
        elgs_y=elgs_y.to("cuda:0")
        



        Q_ni = torch.linalg.pinv(elevecs_x)
        M_ni = torch.linalg.pinv(elevecs_y)



        p121=nn_search(feat_y,feat_x)
    
        p12=self.soft_correspondence(feat_x,feat_y)
   
        for it in range(6):
                  
          
            C_fmapel = Q_ni @ elevecs_y[p121, :]
         
            if self.is_mwp:
                Cel=self.MWP(elgs_x, elgs_y,C_fmapel)

            p126=nn_search(elevecs_y@Cel.t(), elevecs_x) 
         
            clb=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p126,:]
            if self.is_mwp:
                 Clb=self.MWP(gs_x, gs_y,clb)
             
            com_1=torch.cat((evecs_y@ Clb.T,elevecs_y@Cel.T),dim=1)
            com_2=torch.cat((evecs_x, elevecs_x),dim=1)       
            p128 = nn_search(com_1,com_2)
            p121=p128

        loss=self.criterion(evecs_x,p12@evecs_y@Clb.transpose(-2,-1))+self.criterion(elevecs_x,p12@elevecs_y@Cel.transpose(-2,-1))
     


        return loss

    
    def sinkhorn(self, d, sigma=0.1, num_sink=10):
        d = d / d.mean()
        log_p = -d / (2*sigma**2)
        
        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        self.p_adj = torch.exp(log_p).transpose(0, 1)

    def soft_correspondence(self, emb_x,emb_y, sigma = 1e2):
        emb_x = emb_x / emb_x.norm(dim=1, keepdim=True)
        emb_y = emb_y / emb_y.norm(dim=1, keepdim=True)

        D = torch.matmul(emb_x, emb_y.transpose(0, 1))

        
        self.p = torch.nn.functional.softmax(D * sigma, dim=1)
        self.pdj=torch.nn.functional.softmax(D * sigma, dim=0).transpose(0, 1)
        return self.p

    def feat_correspondences(self, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        self.sinkhorn(d)

    def MWP(self, gs_x, gs_y,C):
        # input:
        #   massvec_x/y: [M/N,]
        #   evecs_x/y: [M/N,Kx/Ky]
        #   gs_x/y: [Nf,Kx/Ky]

        # compute MWP functional map
        # C=evecs_x.transpose(-2,-1)@(massvec_x.unsqueeze(-1)*(self.p@evecs_y))
        C_new=torch.zeros_like(C)

        gs_x=gs_x.unsqueeze(-1) # [Nf,Kx]->[Nf,Kx,1]
        gs_y=gs_y.unsqueeze(-1) # [Nf,Ky]->[Nf,Ky,1]

        # MWP filters
        Nf=gs_x.size(0)
        for s in range(Nf):
            C_new+=gs_x[s]*C*gs_y[s].transpose(-2,-1)
        
        return C_new
    


   
    
    def model_test_opt(self,verts_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,elevals_x,elevecs_x,name_x,\
            verts_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y,elevals_y,elevecs_y,name_y):

        feat_x=self.feat_extrac(verts_x, massvec_x, evals=evals_x, evecs=evecs_x, gradX=gradX_x, gradY=gradY_x)
        feat_y=self.feat_extrac(verts_y, massvec_y, evals=evals_y, evecs=evecs_y, gradX=gradX_y, gradY=gradY_y)


        elevals_x=elevals_x[0][:100]
        elevals_y=elevals_y[0][:100]  
        elevecs_x=elevecs_x[:,:100]
        elevecs_y=elevecs_y[:,:100]   

        elevecs_x=elevecs_x.to(dtype=torch.float32)
        elevecs_y=elevecs_y.to(dtype=torch.float32) 

        BT_A_B = elevecs_x.T @ torch.diag(massvec_x) @ elevecs_x
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_x = elevecs_x @ BT_A_B_inv_sqrt

        BT_A_B = elevecs_y.T @ torch.diag(massvec_y) @ elevecs_y
        BT_A_B_inv_sqrt = torch.linalg.pinv(torch.linalg.cholesky(BT_A_B))
        elevecs_y = elevecs_y @ BT_A_B_inv_sqrt

        elgs_x=[]
        positive_evals_x = elevals_x[elevals_x > 0].cpu()
        negative_evals_x = elevals_x[elevals_x < 0].cpu()
        for i in range(len(elevals_x)):
            if elevals_x[i]>0:
                elgs_x.append(Meyer(positive_evals_x[-1],Nf=6)(elevals_x[i].cpu()))
            else:
                elgs_x.append(Meyer(negative_evals_x[-1],Nf=6)(elevals_x[i].cpu())) 
        elgs_x=torch.stack(elgs_x)
        elgs_x=elgs_x.T
        elgs_x=elgs_x.to("cuda:0")

        elgs_y=[]
        positive_evals_y = elevals_y[elevals_y > 0].cpu()
        negative_evals_y = elevals_y[elevals_y < 0].cpu()
        for i in range(len(elevals_y)):
            if elevals_y[i]>0:
                elgs_y.append(Meyer(positive_evals_y[-1],Nf=6)(elevals_y[i].cpu()))
            else:
                elgs_y.append(Meyer(negative_evals_y[-1],Nf=6)(elevals_y[i].cpu())) 
        elgs_y=torch.stack(elgs_y)
        elgs_y=elgs_y.T
        elgs_y=elgs_y.to("cuda:0")
        
        Q_ni = torch.linalg.pinv(elevecs_x)
        M_ni = torch.linalg.pinv(elevecs_y)

        p121=nn_search(feat_y,feat_x)
      
    
        for it in range(6):
            
            C_fmapel = Q_ni @ elevecs_y[p121, :]
            if self.is_mwp:
                Cel=self.MWP(elgs_x, elgs_y,C_fmapel) 
            p126=nn_search(elevecs_y@Cel.t(), elevecs_x) 
            
            clb=evecs_x.transpose(-2,-1)@(torch.diag(massvec_x))@evecs_y[p126,:]
            if self.is_mwp:
                Clb=self.MWP(gs_x, gs_y,clb)

            com_1=torch.cat((evecs_y@ Clb.T,elevecs_y @ Cel.T),dim=1)
            com_2=torch.cat((evecs_x, elevecs_x),dim=1)

            
            p128 = nn_search(com_1,com_2)
            p121=p128

        return p121

  