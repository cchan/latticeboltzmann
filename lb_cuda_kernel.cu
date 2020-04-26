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

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Stream first
    cell_t<float> cell;
    #pragma unroll
    for(int dy = -1; dy <= 1; dy ++) {
        #pragma unroll
        for(int dx = -1; dx <= 1; dx ++) {
            if(y-dy < 0 || y-dy >= N || x-dx < 0 || x-dx >= M)
                cell.d[dy+1][dx+1] = surroundings->d[dy+1][dx+1];
            else
                cell.d[dy+1][dx+1] = cells->d[y-dy][x-dx].d[dy+1][dx+1];
        }
    }

    // Calculate aggregates
    float s1 = cell.d[0][0] + cell.d[0][1] + cell.d[0][2];
    float s2 = cell.d[1][0] + cell.d[1][1] + cell.d[1][2];
    float s3 = cell.d[2][0] + cell.d[2][1] + cell.d[2][2];
    float d = s1 + s2 + s3; // Total density
    float uy = (s3 - s1)/d; // Y component of average velocity
    float ux = (cell.d[0][2] + cell.d[1][2] + cell.d[2][2] - cell.d[0][0] - cell.d[1][0] - cell.d[2][0])/d; // X component of average velocity

    // Display the frame
    float h = atan2f(uy, ux) + PI;
    float s = __saturatef(1000 * sqrtf(ux*ux+uy*uy));
    float v = __saturatef(d);
    frame->d[y][x] = hsv_to_rgb(h, s, v);

    // Collide
    #pragma unroll
    for(int dy = -1; dy <= 1; dy ++) {
        #pragma unroll
        for(int dx = -1; dx <= 1; dx ++) {
            float eu = dy * uy + dx * ux;
            float eq = d * w[dy+1][dx+1] * (1 + r2 * eu + r2*r2/2*eu*eu - r2/2*(ux*ux + uy*uy));

            // Is it blocked? (sizeof bool should == sizeof char)
            char b = (char)blocked->d[y+dy][x+dx];
            // Reflect when blocked, and OMEGA = 1 (no collisions)
            // Otherwise decay toward equilibrium (i.e. collisions) according to OMEGA
            newcells->d[y][x].d[(1-2*b)*dy+1][(1-2*b)*dx+1] = (cell.d[dy+1][dx+1] - eq) * (OMEGA+(1-OMEGA)*b) + eq;
        }
    }
}
}

