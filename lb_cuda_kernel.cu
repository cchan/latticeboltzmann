#include <cuda_runtime_api.h>

#define PI 3.141592653589

// Adapted from https://github.com/hellopatrick/cuda-samples/blob/master/hsv/kernel.cu
__device__ uchar3 hsv_to_rgb(float h, float s, float v) {
    float r, g, b;
    
    float f = h/(PI/3);
    float hi = floorf(f);
    f = f - hi;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t = v * (1 - s * (1 - f));
    
    if(hi == 0.0f || hi == 6.0f) {
        r = v;
        g = t;
        b = p;
    } else if(hi == 1.0f) {
        r = q;
        g = v;
        b = p;
    } else if(hi == 2.0f) {
        r = p;
        g = v;
        b = t;
    } else if(hi == 3.0f) {
        r = p;
        g = q;
        b = v;
    } else if(hi == 4.0f) {
        r = t;
        g = p;
        b = v;
    } else {
        r = v;
        g = p;
        b = q;
    }
    
    unsigned char red = (unsigned char) __float2uint_rn(255.0f * r);
    unsigned char green = (unsigned char) __float2uint_rn(255.0f * g);
    unsigned char blue = (unsigned char) __float2uint_rn(255.0f * b);
    return (uchar3) {red, green, blue};
}

// # Constants for D2Q9 https://arxiv.org/pdf/0908.4520.pdf # Normalized boltzmann distribution (thermal)
// assert(np.all(w == np.flip(w, axis=0)))
// assert(math.isclose(sum(w), 1, rel_tol=1e-6))
#define r2 3
__constant__ const float w[3][3] = {{1.0/36, 1.0/9, 1.0/36},
                                    {1.0/9,  4.0/9, 1.0/9},
                                    {1.0/36, 1.0/9, 1.0/36}};

template<typename T>
struct cell_t {
    T d[3][3];
};

template<typename T>
struct grid_t {
    T d[N][M];
};



extern "C" {
__global__ void fused_collide_stream(grid_t<cell_t<float>>* newcells, grid_t<uchar3>* frame, const grid_t<cell_t<float>>* cells,
                                     const grid_t<bool>* blocked, const cell_t<float>* surroundings) {
    //assert(gridDim.z * blockDim.z == 1);
    //assert(gridDim.y * blockDim.y == N);
    //assert(gridDim.x * blockDim.x == M);

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // look up some hyper optimized convolutional filter cuda impl as a model

    // Calculate aggregates
    cell_t<float> cell = cells->d[y][x];
    float d = 0, uy = 0, ux = 0;
    #pragma unroll
    for(int dy = -1; dy <= 1; dy ++) {
        #pragma unroll
        for(int dx = -1; dx <= 1; dx ++) {
            d += cell.d[dy+1][dx+1];
            uy += cell.d[dy+1][dx+1] * dy;
            ux += cell.d[dy+1][dx+1] * dx;
        }
    }
    uy /= d;
    ux /= d;

    // Display the frame
    float h = atan2f(uy, ux) + PI;
    float s = __saturatef(1000 * sqrtf(ux*ux+uy*uy));
    float v = __saturatef(d);
    frame->d[y][x] = hsv_to_rgb(h, s, v);

    // Collide and stream
    #pragma unroll
    for(int dy = -1; dy <= 1; dy ++) {
        #pragma unroll
        for(int dx = -1; dx <= 1; dx ++) {
            if((y+dy < 0) | (y+dy >= N) | (x+dx < 0) | (x+dx >= M)) {
                // If we're streaming out to surroundings,
                // there must also be an incoming stream from the surroundings.
                newcells->d[y][x].d[-dy+1][-dx+1] = surroundings->d[-dy+1][-dx+1];
            } else {
                float eu = dy * uy + dx * ux;
                float eq = d * w[dy+1][dx+1] * (1 + r2 * eu + r2*r2/2*eu*eu - r2/2*(ux*ux + uy*uy));
                // Decay toward equilibrium, and assign to new cell location
                if(blocked->d[y+dy][x+dx]) {
                    // Reflected because blocked, also OMEGA = 1
                    newcells->d[y+dy][x+dx].d[-dy+1][-dx+1] = cell.d[dy+1][dx+1];
                } else {
                    // Normal
                    newcells->d[y+dy][x+dx].d[ dy+1][ dx+1] = (cell.d[dy+1][dx+1] - eq) * OMEGA + eq;
                }
            }
        }
    }
}
}

