#include "sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to initialize the SRAM manager
void sram_init(SRAM_Manager_t* manager, uint64_t size) {
    manager->sram_size = size;
    manager->used_size = 0;
    manager->allocations = NULL;
    manager->alloc_count = 0;
    manager->alloc_capacity = 0;
}
// Function to allocate memory from SRAM
void* sram_alloc(SRAM_Manager_t* manager, size_t size) {
    if (manager->used_size + size > manager->sram_size) {
        fprintf(stderr, "SRAM allocation failed: Not enough space. (alloc: %lu > %lu)\n", manager->used_size + size, manager->sram_size);
        exit(EXIT_FAILURE);
    }

    // Resize allocations array if necessary
    if (manager->alloc_count >= manager->alloc_capacity) {
        size_t new_capacity = manager->alloc_capacity == 0 ? 1 : manager->alloc_capacity * 2;
        SRAM_Allocation_t* new_allocations = realloc(manager->allocations, new_capacity * sizeof(SRAM_Allocation_t));
        if (!new_allocations) {
            fprintf(stderr, "SRAM allocation failed: Could not resize allocations array.\n");
            exit(EXIT_FAILURE);
        }
        manager->allocations = new_allocations;
        manager->alloc_capacity = new_capacity;
    }

    // Allocate memory
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "SRAM allocation failed: Could not allocate memory.\n");
        exit(EXIT_FAILURE);
    }

    // Record the allocation
    manager->allocations[manager->alloc_count].ptr = ptr;
    manager->allocations[manager->alloc_count].size = size;
    manager->used_size += size;
    manager->alloc_count++;

    return ptr;
}
// Function to free memory allocated from SRAM
void sram_free(SRAM_Manager_t* manager, void* ptr) {
    for (size_t i = 0; i < manager->alloc_count; i++) {
        if (manager->allocations[i].ptr == ptr) {
            free(ptr);
            manager->used_size -= manager->allocations[i].size;

            // Shift remaining allocations
            for (size_t j = i; j < manager->alloc_count - 1; j++) {
                manager->allocations[j] = manager->allocations[j + 1];
            }
            manager->alloc_count--;
            return;
        }
    }
    fprintf(stderr, "SRAM free failed: Pointer not found.\n");
}
// Function to free all SRAM allocations
void sram_free_all(SRAM_Manager_t* manager) {
    for (size_t i = 0; i < manager->alloc_count; i++) {
        free(manager->allocations[i].ptr);
    }
    free(manager->allocations);
    manager->allocations = NULL;
    manager->used_size = 0;
    manager->alloc_count = 0;
    manager->alloc_capacity = 0;
}
// Function to print SRAM usage statistics
void sram_print_stats(SRAM_Manager_t* manager) {
    printf("SRAM Usage Statistics:\n");
    printf("Total Size: %lu bytes\n", manager->sram_size);
    printf("Used Size: %lu bytes\n", manager->used_size);
    printf("Free Size: %lu bytes\n", manager->sram_size - manager->used_size);
    printf("Number of Allocations: %zu\n", manager->alloc_count);
}