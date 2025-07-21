#ifndef SRAM_H
#define SRAM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define macros for configuring SRAM size
#define SRAM_SIZE_1KB 1024  // 1KB
#define SRAM_SIZE_4KB 4096  // 4KB
#define SRAM_SIZE_16KB 16384 // 16KB
#define SRAM_SIZE_64KB 65536 // 64KB
#define SRAM_SIZE_256KB 262144 // 256KB
#define SRAM_SIZE_512KB 524288 // 512KB
#define SRAM_SIZE_1MB 1048576 // 1MB
#define SRAM_SIZE_2MB 2097152 // 2MB
#define SRAM_SIZE_4MB 4194304 // 4MB
#define SRAM_SIZE_8MB 8388608 // 8MB

typedef struct{
    void *ptr;
    size_t size; // Size of the allocated memory in bytes
} SRAM_Allocation_t;

typedef struct {
    uint64_t sram_size;        // SRAM size in bytes
    uint64_t used_size;       // Used SRAM size in bytes
    SRAM_Allocation_t* allocations; // Pointer to an array of SRAM_Allocation
    size_t alloc_count; // Number of allocations in the array
    size_t alloc_capacity; // Capacity of the allocations array
} SRAM_Manager_t;


// Function to initialize the SRAM manager
void sram_init(SRAM_Manager_t* manager, uint64_t size);

// Function to allocate memory from SRAM
void* sram_alloc(SRAM_Manager_t* manager, size_t size);

// Function to free memory allocated from SRAM
void sram_free(SRAM_Manager_t* manager, void* ptr);

// Function to free all SRAM allocations
void sram_free_all(SRAM_Manager_t* manager);

// Function to print SRAM usage statistics
void sram_print_stats(SRAM_Manager_t* manager);

/*
* ============================
* SRAM Configuration (Tiled)
* ============================
*/

    #ifndef SRAM_DEFINE_EXTERN
    #define SRAM_DEFINE_EXTERN

        #define SRAM_DEFAULT_SIZE SRAM_SIZE_8MB

        #define CONV_TILED_OUT_HEIGHT 32  // Height of the tile
        #define CONV_TILED_OUT_WIDTH 32   // Width of the tile
        #define CONV_TILED_OUT_CHANNELS 128  // Number of input channels per tile
        #define CONV_TILED_IN_CHANNELS 64 // Number of output channels per tile

        #define MAXPOOL_TILED_OUT_HEIGHT 32  // Height of the tile
        #define MAXPOOL_TILED_OUT_WIDTH 32   // Width of the tile
        #define MAXPOOL_TILED_CHANNELS 128 // Number of channels per tile

        #define LAYERNORM_TILED_N 1024   // Number of elements per tile

        #define BATCHNORM_TILED_CHANNELS 64 // Number of channels per tile
        #define BATCHNORM_TILED_HEIGHT 64  // Height of the tile
        #define BATCHNORM_TILED_WIDTH 64   // Width of the tile

        #define GEMM_TILED_OUT_DIM 512 // Number of output dimensions per tile
        #define GEMM_TILED_IN_DIM 512 // Number of input dimensions per tile
        #define GEMM_TILED_N 128  // Number of rows per tile

        #define MULTIHEAD_ATTENTION_TILED_Q_LEN 32  // Query length per tile
        #define MULTIHEAD_ATTENTION_TILED_KV_LEN 32  // Key/Value length per tile

        #define ADD_TILED_SIZE 4096 // Number of elements per tile
        #define SIGMOID_TILED_SIZE 4096 // Number of elements per tile

    #endif // SRAM_DEFINE_EXTERN

#endif // SRAM_H