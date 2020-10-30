#ifndef N
#define N 2160
#endif
#ifndef M
#define M 3840
#endif
#ifndef OMEGA
#define OMEGA 0.00000000000001f
#endif
#define PI 3.141592653589f

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
#define r2 3.0f
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

    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    bool isHalo = (x%32 == 0 || x%32 == 31);
    x -= (x/32) * 2 + 1;
    bool isEdge = (x < 0 || x >= M);
    isHalo = isEdge || isHalo;

    cell_t<float> surr = *surroundings;
    cell_t<float> prev = surr, curr = surr, next, newcurr;

    for(int y = 0; y < N + 1; y++) {

    // cell_t<float> asdf;
    // asdf.d[0][0] = 0; asdf.d[0][1] = 0; asdf.d[0][2] = 0;
    // asdf.d[1][0] = 0; asdf.d[1][1] = 0; asdf.d[1][2] = 100;
    // asdf.d[2][0] = 0; asdf.d[2][1] = 0; asdf.d[2][2] = 0;

    // Calculate aggregates
    if(isEdge || y >= N)
        next = surr;
    else
        next = cells->d[y][x];
    float s1 = next.d[0][0] + next.d[0][1] + next.d[0][2];
    float s2 = next.d[1][0] + next.d[1][1] + next.d[1][2];
    float s3 = next.d[2][0] + next.d[2][1] + next.d[2][2];
    float d = s1 + s2 + s3 + 0.0001; // Total density (plus a fudge factor for numerical stability)
                            // Alternative numerical stability method is to prevent any values from going negative.
    float uy = (s3 - s1)/d; // Y component of average velocity
    float ux = (next.d[0][2] + next.d[1][2] + next.d[2][2] - next.d[0][0] - next.d[1][0] - next.d[2][0])/d; // X component of average velocity
    // float mag = uy*uy + ux*ux;
    // uy /= mag;
    // ux /= mag;

    // Display the frame
    if (frame && !isHalo && y > 0) {
        float h = atan2f(uy, ux) + PI;
        float s = __saturatef(1000 * sqrtf(ux*ux+uy*uy));
        float v = __saturatef(d);
        frame->d[y-1][x] = hsv_to_rgb(h, s, v);
    }

    // Compute collide
    if(y > 0 && !blocked->d[y-1][x+1]) {
        float c = 1 - r2/2*(ux*ux + uy*uy);
        #pragma unroll
        for(int dy = 0; dy <= 2; dy ++) {
            #pragma unroll
            for(int dx = 0; dx <= 2; dx ++) {
                float eu = (dy-1) * uy + (dx-1) * ux;
                float eq = d * w[dy][dx] * (c + r2 * eu * (1 + r2/2*eu));
                next.d[dy][dx] = (next.d[dy][dx] - eq) * OMEGA + eq;
            }
        }
    }

    // Exchange adjacent through shuffles
    newcurr.d[0][0] = __shfl_down_sync(0xffffffff, next.d[0][0], 1);
    newcurr.d[0][1] = __shfl_down_sync(0xffffffff, curr.d[0][1], 1);
    newcurr.d[0][2] = __shfl_down_sync(0xffffffff, prev.d[0][2], 1);
    newcurr.d[1][0] = next.d[1][0];
    newcurr.d[1][1] = curr.d[1][1];
    newcurr.d[1][2] = prev.d[1][2];
    newcurr.d[2][0] = __shfl_up_sync(0xffffffff, next.d[2][0], 1);
    newcurr.d[2][1] = __shfl_up_sync(0xffffffff, curr.d[2][1], 1);
    newcurr.d[2][2] = __shfl_up_sync(0xffffffff, prev.d[2][2], 1);

    if(y > 0 && blocked->d[y-1][x]) {
        {
            float tmp = newcurr.d[0][0];
            newcurr.d[0][0] = newcurr.d[2][2];
            newcurr.d[2][2] = tmp;
        }
        {
            float tmp = newcurr.d[0][1];
            newcurr.d[0][1] = newcurr.d[2][1];
            newcurr.d[2][1] = tmp;
        }
        {
            float tmp = newcurr.d[0][2];
            newcurr.d[0][2] = newcurr.d[2][0];
            newcurr.d[2][0] = tmp;
        }
        {
            float tmp = newcurr.d[1][0];
            newcurr.d[1][0] = newcurr.d[1][2];
            newcurr.d[1][2] = tmp;
        }
    }

    if(!isHalo && x >= 0 && x < M && y > 0) {
        newcells->d[y-1][x] = newcurr; // Geez this is 100% of the gap between where we are (2150us) and memory bandwidth bottleneck (500us)
    }
    prev = curr;
    curr = next;
    }
}
}

