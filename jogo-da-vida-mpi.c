#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define N 2048
#define NUM_GER 2000

int NUM_THREADS;


int getNeighbors(int **grid, int i, int j){

    int cima, baixo, esquerda, direita;

    cima = i - 1;
    baixo = i + 1;
    esquerda = j - 1;
    direita = j + 1;

    if(cima < 0) 
        cima = N - 1;

    if(baixo >= N) 
        baixo = 0;

    if(esquerda < 0) 
        esquerda = N - 1;

    if(direita >= N) 
        direita = 0;
    
    return  grid[cima][esquerda] + grid[cima][j] 
            + grid[cima][direita] + grid[i][esquerda]
            + grid[i][direita] + grid[baixo][esquerda]
            + grid[baixo][j] + grid[baixo][direita];     
}

int nextGeneration(int **grid, int **newgrid, int rank){
    int i, j, vizinhos_vivos, total_vivos = 0;
    for (i = (N/NUM_THREADS)*rank; i < (N/NUM_THREADS)*(rank+1); i++){
        for(j = 0; j < N; j++){
            vizinhos_vivos = getNeighbors(grid, i, j);
            if(grid[i][j] == 1 && (vizinhos_vivos == 2 || vizinhos_vivos == 3))
                newgrid[i][j] = 1;       
            else if(grid[i][j] == 0 && vizinhos_vivos == 3)
                newgrid[i][j] = 1;      
            else
            newgrid[i][j] = grid[i][j] * 0;   
            total_vivos += newgrid[i][j]; 
        }
    }
    return total_vivos;
}

int remanescente(int i, int j){
    // define se eu vou passar uma mensagem para a próxima linha
    // ou para a anterior
    int modo = i%j;
    while (modo < 0){
        modo = (j+modo)%j;
    }
    return modo;
}

void message_exchange(int **grid, int rank, int modo, int geracao){
    MPI_Request send_request, recv_request;
    MPI_Status send_status, recv_status;
    int recv_position, send_position, send_rank, recv_rank;
    if(modo == 1){ 
        recv_position = remanescente((N/NUM_THREADS)*rank-1, N);
        send_position = (N/NUM_THREADS)*(rank+1)-1;
        recv_rank = remanescente(rank-1, NUM_THREADS);
        send_rank = remanescente(rank+1, NUM_THREADS);
    } else {
        recv_position = remanescente((N/NUM_THREADS)*(rank+1), N);
        send_position = (N/NUM_THREADS)*rank;
        recv_rank = remanescente(rank+1, NUM_THREADS);
        send_rank = remanescente(rank-1, NUM_THREADS);
    }
    MPI_Isend(grid[send_position], N, MPI_INT, send_rank, geracao*2+1+modo, MPI_COMM_WORLD, &send_request);
    MPI_Irecv(grid[recv_position], N, MPI_INT, recv_rank, geracao*2+1+modo, MPI_COMM_WORLD, &recv_request);
    MPI_Wait(&send_request, &send_status);
    MPI_Wait(&recv_request,&recv_status);
}

int main(){
    MPI_Init(NULL, NULL);
    double time = 0.0;
    int rank, local_alives, global_alives;
    // Inicia matriz
    int **grid = (int **) calloc(N, sizeof(int*));
    int **newgrid = (int **) calloc(N, sizeof(int*));


    for(int i = 0 ; i < N ; i++){
        grid[i] = (int *) calloc(N, sizeof(int));
        newgrid[i] = (int *) calloc(N, sizeof(int));
    }
    //
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_THREADS);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// CONDICOES INICIAIS
    // Building Glider
    int lin = 1, col = 1;
    grid[lin  ][col+1] = 1;
    grid[lin+1][col+2] = 1;
    grid[lin+2][col  ] = 1;
    grid[lin+2][col+1] = 1;
    grid[lin+2][col+2] = 1;


    // Building R-pentomino
    lin =10; col = 30;
    grid[lin][col+1] = 1;
    grid[lin][col+2] = 1;
    grid[lin+1][col  ] = 1;
    grid[lin+1][col+1] = 1;
    grid[lin+2][col+1] = 1;
////

    if(rank == 0){
        time -= MPI_Wtime();
    }

    for(int i = 0; i<NUM_GER; i++){
        local_alives = nextGeneration(grid, newgrid, rank);
        MPI_Reduce(&local_alives, &global_alives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        int **j = grid;
        grid = newgrid;
        newgrid = j;
        if(NUM_THREADS>1){
            message_exchange(grid, rank, 1, i);
            message_exchange(grid, rank, -1, i);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        printf("Células vivas na última geração: %d\n", global_alives);
        time = MPI_Wtime() - time;
        printf("Tempo decorrido na sessão paralela: %f\n", time);
    }
    for(int i = 0; i < N; i++){
        free(grid[i]);
        free(newgrid[i]);
    }
    free(grid);
    free(newgrid);
    MPI_Finalize();
    return 0;
}