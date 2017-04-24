int nnd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *dist1, THCudaTensor *dist2, THCudaLongTensor *idx1, THCudaLongTensor *idx2);


int nnd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *gradxyz1, THCudaTensor *gradxyz2, THCudaTensor *graddist1, THCudaTensor *graddist2, THCudaLongTensor *idx1, THCudaLongTensor *idx2);

