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
    T d[N*M];
};

__device__ int morton(int y, int x) {
    // Interleave lower 16 bits of x and y, so the bits of x
    // are in the even positions and bits from y in the odd;
    // x and y must initially be less than 65536.

    static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};

    y = (y | (y << 8)) & B[3];
    y = (y | (y << 4)) & B[2];
    y = (y | (y << 2)) & B[1];
    y = (y | (y << 1)) & B[0];

    x = (x | (x << 8)) & B[3];
    x = (x | (x << 4)) & B[2];
    x = (x | (x << 2)) & B[1];
    x = (x | (x << 1)) & B[0];

    return x | (y << 1);
}

// NOTE: only dy, dx in {-1, 0, 1} supported!
__device__ int mortonmove(int i, int dy, int dx) {
    return (((i | 0xAAAAAAAA) & 0x55555555 + dx) & 0x55555555)
         | (((i | 0x55555555) & 0xAAAAAAAA + dy) & 0xAAAAAAAA);
}

extern "C" {
__global__ void fused_collide_stream(grid_t<cell_t<float>>* newcells, grid_t<uchar3>* frame, const grid_t<cell_t<float>>* cells,
                                     const grid_t<bool>* blocked, const cell_t<float>* surroundings) {
    //assert(gridDim.z * blockDim.z == 1);
    //assert(gridDim.y * blockDim.y == N);
    //assert(gridDim.x * blockDim.x == M);

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = morton(y, x);

    // Calculate aggregates
    cell_t<float> cell = cells->d[i];
    float s1 = cell.d[0][0] + cell.d[0][1] + cell.d[0][2];
    float s2 = cell.d[1][0] + cell.d[1][1] + cell.d[1][2];
    float s3 = cell.d[2][0] + cell.d[2][1] + cell.d[2][2];
    float d = s1 + s2 + s3; // Total density
    float uy = (s3 - s1)/d; // Y component of average velocity
    float ux = (cell.d[0][2] + cell.d[1][2] + cell.d[2][2] - cell.d[0][0] - cell.d[1][0] - cell.d[2][0])/d; // X component of average velocity

    // Display the frame
    if (frame) {
        float h = atan2f(uy, ux) + PI;
        float s = __saturatef(1000 * sqrtf(ux*ux+uy*uy));
        float v = __saturatef(d);
        frame->d[y * gridDim.x * blockDim.x + x] = hsv_to_rgb(h, s, v);
    }

    // Collide and stream
    int zrow = morton(y - 1, x - 1);
    #pragma unroll
    for(int dy = -1; dy <= 1; dy ++) {
        int znew = zrow;
        #pragma unroll
        for(int dx = -1; dx <= 1; dx ++) {
            if((y+dy < 0) | (y+dy >= N) | (x+dx < 0) | (x+dx >= M)) {
                // If we're streaming out to surroundings,
                // there must also be an incoming stream from the surroundings.
                newcells->d[i].d[-dy+1][-dx+1] = surroundings->d[-dy+1][-dx+1];
            } else {
                float eu = dy * uy + dx * ux;
                float eq = d * w[dy+1][dx+1] * (1 + r2 * eu + r2*r2/2*eu*eu - r2/2*(ux*ux + uy*uy));
                // Decay toward equilibrium, and assign to new cell location
                if(blocked->d[znew]) {
                    // Reflected because blocked, also OMEGA = 1
                    newcells->d[znew].d[-dy+1][-dx+1] = cell.d[dy+1][dx+1];
                } else {
                    // Normal
                    newcells->d[znew].d[ dy+1][ dx+1] = (cell.d[dy+1][dx+1] - eq) * OMEGA + eq;
                }
            }
            znew = (((znew | 0b10101010) + 1) & 0b01010101) | (znew & 0b10101010);
        }
        zrow = (((zrow | 0b01010101) + 1) & 0b10101010) | (zrow & 0b01010101);
    }
}
}

