#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void input_vector(int *vec, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % 100;
        }
    }
}

void multiply_mat_vec_blocks(int *local_mat, int *local_vec, int *local_res, int cols, int start_row, int end_row, int start_col, int end_col) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = start_col; j < end_col; j++) {
            local_res[i] += local_mat[i * cols + j] * local_vec[j];
        }
    }
}

int main(int argc, char *argv[]) {
    int comm_sz, my_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int vec_size = cols;
    int res_size = rows;
    int block_dim = (int)sqrtl(comm_sz);
    int blocks_total = block_dim * block_dim;
    int block_rows = rows / block_dim;
    int block_cols = cols / block_dim;

    int *vec = malloc(vec_size * sizeof(int));
    int *mat = malloc(rows * cols * sizeof(int));
    int *result = NULL;
    int *local_res_vec = malloc(res_size * sizeof(int));

    for (int i = 0; i < res_size; i++) {
        local_res_vec[i] = 0;
    }

    if (my_rank == 0) {
        result = malloc(res_size * sizeof(int));
        for (int i = 0; i < res_size; i++) {
            result[i] = 0;
        }
    }
    input_vector(vec, vec_size, my_rank);
    input_vector(mat, rows * cols, my_rank);
    MPI_Bcast(vec, vec_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

    int start_row = (my_rank / block_dim) * block_rows;
    int end_row = start_row + block_rows;
    int start_col = (my_rank % block_dim) * block_cols;
    int end_col = start_col + block_cols;
    if (my_rank < blocks_total) {
        if (my_rank == blocks_total - 1) {
            end_row = rows;
            end_col = cols;
        } else if ((my_rank + 1) % block_dim == 0) {
            end_col = cols;
        } else if ((my_rank + 1) > blocks_total - block_dim) {
            end_row = rows;
        }
    }

    double start_time = MPI_Wtime();
    if (my_rank < blocks_total) {
        multiply_mat_vec_blocks(mat, vec, local_res_vec, cols, start_row, end_row, start_col, end_col);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(local_res_vec, result, res_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double duration = end_time - start_time;
    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("time taken: %lf\n", max_duration * 1e3);
    }
    free(mat);
    free(result);
    free(local_res_vec);
    free(vec);

    MPI_Finalize();
    return 0;
}