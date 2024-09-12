import torch
import torch.nn.functional as F
from structlog import get_logger
import pickle
import numpy as np
from scipy.linalg import block_diag
import time
import openfhe
from openfhe import SecretKeyDist, ScalingTechnique, CCParamsCKKSRNS, SecurityLevel, FHECKKSRNS, GenCryptoContext, PKESchemeFeature


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class enc_GenHECNN():
    
    def __init__(self, cnn, cc, key_pair, input_dim):
        
        self.cc = cc
        self.key_pair = key_pair
        
        self.kernel_size = cnn.conv_CNV.kernel_size[0]
        self.stride = cnn.conv_CNV.stride[0]
        self.n_kernels = cnn.conv_CNV.weight.shape[0]
        
        self.conv_CNV_bias = cnn.conv_CNV.bias.detach().numpy()
        self.conv_CNV_weights = cnn.conv_CNV.weight.data.view(cnn.conv_CNV.out_channels,
                                                              cnn.conv_CNV.kernel_size[0]).numpy()
        
        self.conv_SNV_bias = cnn.conv_SNV.bias.detach().numpy()
        self.conv_SNV_weights = cnn.conv_SNV.weight.data.view(cnn.conv_SNV.out_channels,
                                                              cnn.conv_SNV.kernel_size[0]).numpy()
        
        self.conv_dim = [int((input_dim[0] - self.conv_CNV_weights.shape[1])//self.stride + 1),
                         int((input_dim[1] - self.conv_SNV_weights.shape[1])//self.stride + 1)]
        
        self.encConvCNVWeights = []
        self.encConvCNVBias = []
        
        self.encConvSNVWeights = []
        self.encConvSNVBias = []
        
        self.linear_weights = cnn.fc1.weight.T.data.numpy()
        self.linear_bias = cnn.fc1.bias.detach().numpy()
        
        self.output_dim = next_power_of_2(self.linear_weights.shape[1])
        self.eval_sum_col_keys = cc.EvalSumColsKeyGen(key_pair[1])
        self.eval_sum_row_keys = self.cc.EvalSumRowsKeyGen(key_pair[1], rowSize = self.output_dim)
        
        self.encLinearCNVWeights = []
        self.encLinearSNVWeights = []
        
        self.encLinearBias = []
        
        self.mask = []
        
    def dataEncoding(self, data, mode):
        
        enc_data = []    
        for i in range(self.conv_dim[mode]):
            enc_data.append(np.pad(data[(i*self.stride):(i*self.stride+self.kernel_size)], (0, self.output_dim - self.kernel_size), 'constant', constant_values=(0)))

        enc_data = np.concatenate(enc_data, axis=0)
        
        return enc_data
    
    def __encode4conv1D(self, W, mode):
        enc_W = np.tile(np.pad(W, (0, self.output_dim - W.size), 'constant', constant_values=(0)), self.conv_dim[mode])
        return enc_W
    
    def __encode4linear(self, W):
        W = np.pad(W, ((0, 0), (0, self.output_dim - W.shape[1])), 'constant', constant_values=(0))
        enc_W = W.flatten()        
        return enc_W
    
    def __encodeConv(self):
        for i in range(self.n_kernels):
            self.encConvCNVWeights.append(self.cc.MakeCKKSPackedPlaintext(self.__encode4conv1D(self.conv_CNV_weights[i], 0)))
            self.encConvCNVBias.append(self.cc.MakeCKKSPackedPlaintext(np.tile(self.conv_CNV_bias[i], self.output_dim*self.conv_dim[0])))
            self.encConvSNVWeights.append(self.cc.MakeCKKSPackedPlaintext(self.__encode4conv1D(self.conv_SNV_weights[i], 1)))
            self.encConvSNVBias.append(self.cc.MakeCKKSPackedPlaintext(np.tile(self.conv_SNV_bias[i], self.output_dim*self.conv_dim[1])))

    def __encodeLinear(self):        
        for i in range(self.n_kernels):
            self.encLinearCNVWeights.append(self.cc.MakeCKKSPackedPlaintext(self.__encode4linear(self.linear_weights[i*(self.conv_dim[0] + self.conv_dim[1]):((i+1)*(self.conv_dim[0]) + i*self.conv_dim[1]), :])))    
            self.encLinearSNVWeights.append(self.cc.MakeCKKSPackedPlaintext(self.__encode4linear(self.linear_weights[((i+1)*(self.conv_dim[0]) + i*self.conv_dim[1]):(i+1)*(self.conv_dim[0] + self.conv_dim[1]), :])))
        
        self.encLinearBias = self.cc.MakeCKKSPackedPlaintext(np.tile(np.pad(self.linear_bias, (0, self.output_dim - self.linear_bias.size), 'constant', constant_values=(0)), self.output_dim))
    
    def __createMask(self):
        self.mask = np.zeros(int((2**15)/2))
        self.mask[:self.output_dim] = 1
        self.mask = self.cc.MakeCKKSPackedPlaintext(self.mask)
    
    def encodeModel(self):
        self.__encodeConv()
        self.__encodeLinear()
        self.__createMask()
    
    def encryptModel(self):
        for i in range(self.n_kernels):
            self.encConvCNVWeights[i] = self.cc.Encrypt(self.key_pair[0], self.encConvCNVWeights[i])
            self.encConvCNVBias[i] = self.cc.Encrypt(self.key_pair[0], self.encConvCNVBias[i])
            
            self.encConvSNVWeights[i] = self.cc.Encrypt(self.key_pair[0], self.encConvSNVWeights[i])
            self.encConvSNVBias[i] = self.cc.Encrypt(self.key_pair[0], self.encConvSNVBias[i])
            
            self.encLinearCNVWeights[i] = self.cc.Encrypt(self.key_pair[0], self.encLinearCNVWeights[i])
            self.encLinearSNVWeights[i] = self.cc.Encrypt(self.key_pair[0], self.encLinearSNVWeights[i])
        
        self.encLinearBias = self.cc.Encrypt(self.key_pair[0], self.encLinearBias)
    
    def forward(self, x_CNV, x_SNV):
        for i in range(self.n_kernels):
            ### CONV1D LAYER - CNV          -->         1 MultDepth
            conv_CNV = self.cc.EvalMult(x_CNV, self.encConvCNVWeights[i])
            conv_CNV = self.cc.EvalSumCols(conv_CNV, self.output_dim, self.eval_sum_col_keys)
            conv_CNV = self.cc.EvalAdd(conv_CNV, self.encConvCNVBias[i])
            
            ### CONV1D LAYER - SNV          -->         1 MultDepth
            conv_SNV = self.cc.EvalMult(x_SNV, self.encConvSNVWeights[i])
            conv_SNV = self.cc.EvalSumCols(conv_SNV, self.output_dim, self.eval_sum_col_keys)
            conv_SNV = self.cc.EvalAdd(conv_SNV, self.encConvSNVBias[i])
            
            ### ACTIVATION LAYER - SQUARE   -->         2 MultDepth
            conv_CNV = self.cc.EvalMult(conv_CNV, conv_CNV)
            conv_SNV = self.cc.EvalMult(conv_SNV, conv_SNV)
            
            ### LINEAR LAYER - CNV          -->         3 MultDepth
            linear_CNV = self.cc.EvalMult(conv_CNV, self.encLinearCNVWeights[i])    
            linear_CNV = self.cc.EvalSumRows(linear_CNV, self.output_dim, self.eval_sum_row_keys)
            
            ### LINEAR LAYER - SNV          -->         3 MultDepth
            linear_SNV = self.cc.EvalMult(conv_SNV, self.encLinearSNVWeights[i])    
            linear_SNV = self.cc.EvalSumRows(linear_SNV, self.output_dim, self.eval_sum_row_keys)

            if i == 0:
                ### Results unification     -->         3 MultDepth
                ct_output = self.cc.EvalAdd(linear_CNV, linear_SNV)
            else:
                ### Results unification     -->         3 MultDepth
                ct_output_temp = self.cc.EvalAdd(linear_CNV, linear_SNV)
                ct_output = self.cc.EvalAdd(ct_output, ct_output_temp)

        ct_output = self.cc.EvalAdd(ct_output, self.encLinearBias)
        return self.cc.EvalMult(ct_output, self.mask)
    
