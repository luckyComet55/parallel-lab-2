#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4096
#define MAX_FILE_NAME sizeof("Performance/4096/32")

void fillMatrix(int **A) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                A[i][j] = i + SIZE;
            } else {
                A[i][j] = i;
                A[j][i] = i;
            }
        }
    }
}

void allocateMatrix(int ***mat, int size) {
    // Allocate rows*cols contiguous items
    int *p = (int *)malloc(sizeof(int *) * size * size);

    // Allocate row pointers
    *mat = (int **)malloc(size * sizeof(int *));

    // Set up the pointers into the contiguous memory
    for (int i = 0; i < size; i++) {
        (*mat)[i] = &(p[i * size]);
    }
}

int freeMatrix(int ***mat) {
    free(&((*mat)[0][0]));
    free(*mat);
    return 0;
}

void multiplyMatByMat(int **a, int **b, int size, int ***c) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int val = 0;
            for (int k = 0; k < size; ++k) {
                val += a[i][k] * b[k][j];
            }
            (*c)[i][j] = val;
        }
    }
}

int main(int argc, char *argv[]) {
    int **A = NULL, **B = NULL, **C = NULL;
    int count = SIZE * SIZE;

    MPI_Comm procGrid;
    int procNum;
    int sqrtProcNum;
    int blockNum;
    int bCastData[2], coord[2];

    int **locA = NULL, **locB = NULL, **locC = NULL;
    int left, right, up, down;

    clock_t start, end;

    start = clock();
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int n;

        sqrtProcNum = (int)sqrt(procNum);
        blockNum = SIZE / (int)sqrt(procNum);

        allocateMatrix(&A, SIZE);
        allocateMatrix(&B, SIZE);

        fillMatrix(A);
        fillMatrix(B);

        allocateMatrix(&C, SIZE);

        bCastData[0] = sqrtProcNum;
        bCastData[1] = blockNum;
    }

    MPI_Bcast(&bCastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
    sqrtProcNum = bCastData[0];
    blockNum = bCastData[1];

    int dim[2] = {sqrtProcNum, sqrtProcNum};
    int period[2] = {1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &procGrid);

    int originalMatSize[2] = {SIZE, SIZE};
    int subMatSize[2] = {blockNum, blockNum};
    int startInd[2] = {0, 0};
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, originalMatSize, subMatSize, startInd, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, blockNum * sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *globalptrA = NULL;
    int *globalptrB = NULL;
    int *globalptrC = NULL;
    if (rank == 0) {
        globalptrA = &(A[0][0]);
        globalptrB = &(B[0][0]);
        globalptrC = &(C[0][0]);
    }

    int *sendNum = (int *)malloc(sizeof(int) * procNum);
    int *displacements = (int *)malloc(sizeof(int) * procNum);

    if (rank == 0) {
        for (int i = 0; i < procNum; i++) {
            sendNum[i] = 1;
        }

        int disp = 0;
        for (int i = 0; i < sqrtProcNum; i++) {
            for (int j = 0; j < sqrtProcNum; j++) {
                displacements[i * sqrtProcNum + j] = disp;
                disp++;
            }
            disp += (blockNum - 1) * sqrtProcNum;
        }
    }

    allocateMatrix(&locA, blockNum);
    allocateMatrix(&locB, blockNum);

    MPI_Scatterv(globalptrA, sendNum, displacements, subarrtype, &(locA[0][0]), count / (procNum), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalptrB, sendNum, displacements, subarrtype, &(locB[0][0]), count / (procNum), MPI_INT, 0, MPI_COMM_WORLD);

    allocateMatrix(&locC, blockNum);

    for (int i = 0; i < blockNum; i++) {
        for (int j = 0; j < blockNum; j++) {
            locC[i][j] = 0;
        }
    }

    MPI_Cart_coords(procGrid, rank, 2, coord);

    MPI_Cart_shift(procGrid, 1, coord[0], &left, &right);
    MPI_Sendrecv_replace(&(locA[0][0]), blockNum * blockNum, MPI_INT, left, 1, right, 1, procGrid, MPI_STATUS_IGNORE);

    MPI_Cart_shift(procGrid, 0, coord[1], &up, &down);
    MPI_Sendrecv_replace(&(locB[0][0]), blockNum * blockNum, MPI_INT, up, 1, down, 1, procGrid, MPI_STATUS_IGNORE);

    int **multiplyRes = NULL;
    allocateMatrix(&multiplyRes, blockNum);

    for (int k = 0; k < sqrtProcNum; k++) {
        multiplyMatByMat(locA, locB, blockNum, &multiplyRes);

        for (int i = 0; i < blockNum; i++) {
            for (int j = 0; j < blockNum; j++) {
                locC[i][j] += multiplyRes[i][j];
            }
        }

        MPI_Cart_shift(procGrid, 1, 1, &left, &right);
        MPI_Sendrecv_replace(&(locA[0][0]), blockNum * blockNum, MPI_INT, left, 1, right, 1, procGrid, MPI_STATUS_IGNORE);

        MPI_Cart_shift(procGrid, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(locB[0][0]), blockNum * blockNum, MPI_INT, up, 1, down, 1, procGrid, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(&(locC[0][0]), count / procNum, MPI_INT, globalptrC, sendNum, displacements, subarrtype, 0, MPI_COMM_WORLD);

    freeMatrix(&locC);
    freeMatrix(&multiplyRes);

    MPI_Finalize();
    end = clock();

    if (rank == 0) {
        printf("time taken: %lf\n", (double)end - start);
    }

    return 0;
}