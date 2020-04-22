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
__global__ void fused_collide_stream(grid_t<cell_t<float>>* newcells, const grid_t<cell_t<float>>* cells,
                                     const grid_t<bool>* blocked, const cell_t<float>* surroundings) {
    //assert(gridDim.z * blockDim.z == 1);
    //assert(gridDim.y * blockDim.y == N);
    //assert(gridDim.x * blockDim.x == M);

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // look up some hyper optimized convolutional filter cuda impl as a model

    // Collide
    cell_t<float> cell = cells->d[y][x];
    float d = 0, uy = 0, ux = 0;
    for(int dy = -1; dy <= 1; dy ++) {
        for(int dx = -1; dx <= 1; dx ++) {
            d += cell.d[dy+1][dx+1];
            uy += cell.d[dy+1][dx+1] * dy;
            ux += cell.d[dy+1][dx+1] * dx;
        }
    }
    uy /= d;
    ux /= d;
    for(int dy = -1; dy <= 1; dy ++) {
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
