nvcc -keep -cubin --use_fast_math -O3 -Xptxas -O3,-v -arch sm_75 --extra-device-vectorization --restrict lb_cuda_kernel.cu && cuobjdump -sass lb_cuda_kernel.cubin | grep '\/\*0' > lb_cuda_kernel.sass
