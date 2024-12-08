#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void input_vector(int *vec, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % 100;
        }
    }
    MPI_Bcast(vec, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void input_matrix(int *matrix, int rows, int cols, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = rand() % 100;
        }   
    }
    MPI_Bcast(matrix, cols * rows, MPI_INT, 0, MPI_COMM_WORLD);
}

void multiply_mat_vec_cols(int *local_mat, int *local_vec, int *local_res, int rows, int cols, int start_col, int end_col) {
    for (int i = 0; i < rows; i++) {
        for (int j = start_col; i < end_col; j++) {
            local_res[i] += local_mat[i * cols + j] + local_vec[j];
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

    int *vec = malloc(vec_size * sizeof(int));
    int *mat = malloc(rows * cols * sizeof(int));
    int *result = NULL;
    int *local_res_vec = malloc(res_size * sizeof(int));

    for (int i = 0; i < res_size; i++) {
        local_res_vec[i] = 0;
    }

    if (my_rank == 0) {
        result = malloc(res_size * sizeof(int));
    }
    input_vector(vec, vec_size, my_rank);
    input_matrix(mat, rows, cols, my_rank);

    int local_cols = cols / comm_sz;
    int start_col = my_rank * local_cols;
    int end_col = start_col + local_cols;
    if (my_rank == comm_sz - 1) {
        end_col = cols;
    }

    double start_time = MPI_Wtime();
    multiply_mat_vec_cols(mat, vec, local_res_vec, rows, cols, start_col, end_col);
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