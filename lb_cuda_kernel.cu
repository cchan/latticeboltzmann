#ifndef N
#define N 2160
#endif
#ifndef M
#define M 3840
#endif
#ifndef INNER_TIMESTEPS
#define INNER_TIMESTEPS 1
#endif
#ifndef INNER_BLOCK
#define INNER_BLOCK 2000000
#endif
#ifndef OMEGA
#define OMEGA 0.00000000000001f
#endif
#define PI 3.141592653589f

#include "cuda_fp16.h"

#ifdef half_enable
#define FP half
#else
#define FP float
#endif

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

template<typename T>
__device__ __forceinline__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

__device__ __forceinline__ void prefetch_l1 (unsigned int addr)
{
  asm volatile(" prefetch.global.L1 [ %1 ];": "=r"(addr) : "r"(addr));
}

__device__ __forceinline__ void prefetch_l2 (unsigned int addr)
{
  asm volatile(" prefetch.global.L2 [ %1 ];": "=r"(addr) : "r"(addr));
}

#ifdef USE_PACKED_CELLS
// Something to try......
// each cell is 9 * 4 bytes, which doesn't align well (you can load up to 16B in one insn)
// and also probably has a lot of shared info (i.e. shared-exponent floating point).
// This means that a custom binary format (instead of cell_t<half>) could be used for global memory,
// and then manually shifted into IEEE fp32 by h2f.
    // 8b exponent + 20b mantissa * 9 = 188b < 192b = 24B. 1 cell = 3 * LDG.E.64
    // ... could also just use fixed point (i.e. manipulate integers and assume some fixed exponent like 2**-8) lol
// This would necessitate more subtle comparison testing because it won't be as obviously wrong as fp16 was (which has 10 bits of mantissa).
// Would also be worth doing a validation against an analytical solution.

struct packed_cell {
    int2 d[3];
};

__device__ __forceinline__ cell_t<float> p2f(const packed_cell& p) {
    // assumes the sign is always positive... which it should be.
    cell_t<float> c;
    int mantissa = ((*(int*)&p.d[0]) & 255) << 23;
    return c;
}
__device__ __forceinline__ packed_cell f2p(const cell_t<float>& c) {
    // extract all the components
    // take the maximum exponent we can find
    // for each of the 9 components, rightshift the mantissa by (max - mine) to make it fixed point
    // then rightshift it again by 3 for the known precision reduction to 20b, which is still way better than the 10b of fp16

    packed_cell p;
    int maxexp = 0;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            int exp1 = ((*(int*)&c.d[i][j]) >> 23) & 255;
            maxexp = max(maxexp, exp1);
        }
    }
    copy_to_bit_offset(&packed_cell, maxexp, 10);
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            int exp1 = ((*(int*)&c.d[i][j]) >> 23) & 255;
            int mantissa = ((*(int*)&c.d[i][j]) & ((1 << 23) - 1)) >> (maxexp - exp1 + 3);
        }
    }
    ((unsigned char*)&packed_cell)
    return p;
}
#endif // USE_PACKED_CELLS

#ifdef half_enable
__device__ __forceinline__ cell_t<float> h2f(const cell_t<half>& c) {
    cell_t<float> c2;
    #pragma unroll
    for(int i = 0; i <= 2; i ++) {
        #pragma unroll
        for(int j = 0; j <= 2; j ++) {
            c2.d[i][j] = c.d[i][j];
        }
    }
    return c2;
}

__device__ __forceinline__ cell_t<half> f2h(const cell_t<float>& c) {
    cell_t<half> c2;
    #pragma unroll
    for(int i = 0; i <= 2; i ++) {
        #pragma unroll
        for(int j = 0; j <= 2; j ++) {
            c2.d[i][j] = c.d[i][j];
        }
    }
    return c2;
}
#else
#define h2f(x) (x)
#define f2h(x) (x)
#endif

// Did not work :(
__device__ __forceinline__ cell_t<float> __ldcs_cell(const cell_t<float>* cell) {
    cell_t<float> c;
    // also tried ldg, ldca
    c.d[0][0] = __ldcs(((float*)cell) + 0);
    c.d[0][1] = __ldcs(((float*)cell) + 1);
    c.d[0][2] = __ldcs(((float*)cell) + 2);
    c.d[1][0] = __ldcs(((float*)cell) + 3);
    c.d[1][1] = __ldcs(((float*)cell) + 4);
    c.d[1][2] = __ldcs(((float*)cell) + 5);
    c.d[2][0] = __ldcs(((float*)cell) + 6);
    c.d[2][1] = __ldcs(((float*)cell) + 7);
    c.d[2][2] = __ldcs(((float*)cell) + 8);
    return c;
}

__device__ __forceinline__ void __stcs_cell(const cell_t<float>* cell, const cell_t<float>& c) {
    __stcs(((float*)cell) + 0, c.d[0][0]);
    __stcs(((float*)cell) + 1, c.d[0][1]);
    __stcs(((float*)cell) + 2, c.d[0][2]);
    __stcs(((float*)cell) + 3, c.d[1][0]);
    __stcs(((float*)cell) + 4, c.d[1][1]);
    __stcs(((float*)cell) + 5, c.d[1][2]);
    __stcs(((float*)cell) + 6, c.d[2][0]);
    __stcs(((float*)cell) + 7, c.d[2][1]);
    __stcs(((float*)cell) + 8, c.d[2][2]);
}

template<bool shouldDisplay>
__device__ void fcs(grid_t<cell_t<FP>>* newcells, grid_t<uchar3>* frame, const grid_t<cell_t<FP>>* cells,
                                     const grid_t<bool>* blocked, const cell_t<FP>* surroundings) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    bool isHalo = (x%32 < INNER_TIMESTEPS || x%32 > 31 - INNER_TIMESTEPS);
    x -= (x/32) * INNER_TIMESTEPS * 2 + INNER_TIMESTEPS;
    bool isEdge = (x < 0 || x >= M);
    isHalo = isEdge || isHalo;

    cell_t<float> surr = h2f(*surroundings);
    cell_t<float> prev, curr, next;

    // This fails because there is far too little shared memory and also I think we're already using L1/L2
    // (at the same INNER_BLOCK and INNER_TIMESTEPS it's actually slower to use shared than just directly accessing newcells[][]).
    // __shared__ cell_t<FP> cache[32][INNER_BLOCK + 2 * INNER_TIMESTEPS + 1];
    // If it's already making use of L1/L2, why is it so slow then? Especially on the multi-iteration runs.

    for(int z = -1; z < N; z += INNER_BLOCK) {
    if(z == -1) {
        prev = surr;
        curr = surr;
    } else {
        prev = cells->d[z-1][x]; // technically incorrect if you're incrementing z by 1 in the big loop
        curr = cells->d[z][x];
    }

    for(int timestep = 0; timestep < INNER_TIMESTEPS; timestep ++) {

    for(int y = z; y < z + INNER_BLOCK + 2 * INNER_TIMESTEPS && y < N; y++) {
        // if(y&31==0)
        //     prefetch_l2((unsigned int)&cells->d[y+32][x]); // This produces no appreciable benefit. :(
        // also tried using "nextnext" to get it loaded into registers on the previous iteration, but that didn't seem to help
        // why wouldn't it hit memory bandwidth then??? is it actually blocked by the latency (32ish cycles?) of loads from l1 to registers?
        // it could be legitimately compute bottlenecked... but that seems so unlikely given that other sims were able to hit mem bandwidth. Diff arch tho.
        // no, core overclocking does literally nothing and mem overclocking is basically linear.
        // ah possibly because it's unaligned / not using LDG.E.128.SYS aligned vectorized loads!
        // An attempt was previously made to simply expand the struct to 4 floats per row to use only aligned loads but that is -40% performance.
        // Could possibly interleave 9 128-bit loads, in 1024-bit coalesced chunks, to do four rows at once.
        // 1111 ... x32
        // 1111
        // 1222
        // 2222
        // 2233
        // 3333
        // 3334
        // 4444
        // 4444
        // so every memory load instruction is 1) LDG.E.128.SYS (float4) 2) aligned 3) coalesced
        // struct coalescing_block {
        //     float4 a[32], b[32], c[32], d[32], e[32], f[32], g[32], h[32], i[32];
        // }
        // Actually a better way to do this might be to have 9 separate arrays (SoA), each containing the direction, and load 4 at a time from each of those.
            // Register pressure though... 36 registers just from those loads. Maybe SoA is not the right way to go.
        // It's possible that the above doesn't increase register pressure if done carefully (only 3 128bit registers in use at a time, reuse the earlier registers) but that takes a lot of work.
        // I should probably just implement 3D lattice boltzmann first before attempting this micro-optimization, since we're already at 92% achievable.
        //
        // NOTE: According to the Occupancy Calculator the ideal register use is < 64 per thread. Since we are currently at 57, using any more will affect maximum occupancy.
        // An attempt to get some of the useless elements of prev out of registers didn't do anything:
            // prev0 = curr.d[0][2];
            // prev1 = curr.d[1][2];
            // prev2 = curr.d[2][2];
        //
        // NOTE 2: to circumvent the memory bottleneck, we need to do more compute per unit of data. That means maybe do 2 or more iterations of LBM on a given block. This increases halo size though.
            // But this doesn't have to be at the individual warp level - we can have larger chunks that fit into (half of?) L2 cache, which has 5x bandwidth, and iterate enough times (5?) to balance with global memory.
            // Just implement 3D first though, because all these considerations and tunings dramatically change.
        // NOTE 3: can we make it 32-wide instead of 30-wide? because 30-wide hobbles 3D. but the edge ones will have to read extra, stalling the others...
        // NOTE 4: register reuse caches and register banks/ports - see https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9839-discovering-the-turing-t4-gpu-architecture-with-microbenchmarks.pdf manual sass tuning

        // 50ish rows fit into L1
        // 80ish rows fit into L2
        // so lookahead to 20 while loading the next 20?? just test out some lookaheads for the inner inner loop.

        // Calculate aggregates
        if(isEdge || y < -1 || y >= N - 1)
            next = surr;
        else if(timestep == 0)
            next = h2f(cells->d[y+1][x]);
            // next = h2f(__ldcs_cell(&cells->d[y+1][x]));
        else
            next = h2f(newcells->d[y+1][x]);
        float s1 = next.d[0][0] + next.d[0][1] + next.d[0][2];
        float s2 = next.d[1][0] + next.d[1][1] + next.d[1][2];
        float s3 = next.d[2][0] + next.d[2][1] + next.d[2][2];
        float d = s1 + s2 + s3 + 0.0001; // Total density (plus a fudge factor for numerical stability)
                                         // Alternative numerical stability method is to prevent any values from going negative, or otherwise normalize.
        // Adding 10 floating point multiplies here kills performance by ~50%
        float uy = (s3 - s1)/d; // Y component of average velocity
        float ux = (next.d[0][2] + next.d[1][2] + next.d[2][2] - next.d[0][0] - next.d[1][0] - next.d[2][0])/d; // X component of average velocity

        if(y == -1) {
            prev = curr;
            curr = next;
            continue;
        }

        if constexpr(shouldDisplay) {
            // Display the frame
            if (timestep == INNER_TIMESTEPS - 1 && frame && !isHalo) {
                float h = atan2f(uy, ux) + PI;
                float s = __saturatef(1000 * sqrtf(ux*ux+uy*uy));
                float v = __saturatef(d);
                if(blocked->d[y][x]) {
                    h = s = v = 0;
                }
                frame->d[y][x] = hsv_to_rgb(h, s, v);
            }
        }

        // Compute collide (if blocked, don't do this - no thermal perturbation, purely reflection)
        if(!blocked->d[y][x]) {
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
        cell_t<float> newcurr;
        newcurr.d[0][0] = __shfl_down_sync(0xffffffff, next.d[0][0], 1);
        newcurr.d[0][1] = __shfl_down_sync(0xffffffff, curr.d[0][1], 1);
        newcurr.d[0][2] = __shfl_down_sync(0xffffffff, prev.d[0][2], 1);
        newcurr.d[1][0] = next.d[1][0];
        newcurr.d[1][1] = curr.d[1][1];
        newcurr.d[1][2] = prev.d[1][2];
        newcurr.d[2][0] = __shfl_up_sync(0xffffffff, next.d[2][0], 1);
        newcurr.d[2][1] = __shfl_up_sync(0xffffffff, curr.d[2][1], 1);
        newcurr.d[2][2] = __shfl_up_sync(0xffffffff, prev.d[2][2], 1);

        prev = curr;
        curr = next;

        if(!isEdge) {
            // Reflect the new cell if blocked
            if(blocked->d[y][x]) {
                swap(newcurr.d[0][0], newcurr.d[2][2]);
                swap(newcurr.d[0][1], newcurr.d[2][1]);
                swap(newcurr.d[0][2], newcurr.d[2][0]);
                swap(newcurr.d[1][0], newcurr.d[1][2]);
            }

            // Write the new cell if not a halo cell
            if(!isHalo && (y > z + INNER_TIMESTEPS || z == -1)) {
                newcells->d[y][x] = f2h(newcurr);
                // __stcs_cell(&newcells->d[y][x], f2h(newcurr));
            }
        }
    }

    }

    }
}
extern "C" {
    __global__ void fused_collide_stream(grid_t<cell_t<FP>>* newcells, grid_t<uchar3>* frame, const grid_t<cell_t<FP>>* cells,
        const grid_t<bool>* blocked, const cell_t<FP>* surroundings) {
        if(frame)
            fcs<true>(newcells, frame, cells, blocked, surroundings);
        else
            fcs<false>(newcells, frame, cells, blocked, surroundings);
    }
}

