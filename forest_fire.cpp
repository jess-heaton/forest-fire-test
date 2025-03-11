#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

// Define states for our grid
enum { EMPTY = 0, TREE = 1, BURNING = 2, DEAD = 3 };

// Generate a random grid of size N x N.
void generate_random_grid(int N, double p, vector<int>& grid) {
    grid.resize(N * N, EMPTY);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            double r = (double) rand() / RAND_MAX;
            if (r < p) {
                grid[i * N + j] = TREE;
            }
        }
    }
}

// Read a grid from a file.
void read_grid(const string& filename, vector<int>& grid, int& N) {
    ifstream infile(filename.c_str());
    infile >> N;
    grid.resize(N * N, 0);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            infile >> grid[i * N + j];
        }
    }
    infile.close();
}

// Write the grid to a file.
void write_grid(const string& filename, const vector<int>& grid, int N) {
    ofstream outfile(filename.c_str());
    outfile << N << endl;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            outfile << grid[i * N + j] << " ";
        }
        outfile << endl;
    }
    outfile.close();
}

// Divide the grid rows among MPI tasks
void distribute_grid(int N, int iproc, int nproc, int &i0, int &i1) {
    int rows = N / nproc;
    i0 = iproc * rows;
    // last rank takes any remainder
    if (iproc == nproc - 1)
        i1 = N;
    else
        i1 = i0 + rows;
}

// Perform the forest fire simulation
void simulate_fire(int N, vector<int>& globalGrid, int iproc, int nproc, 
                   int& steps, bool &fireReachedBottom) 
{
    bool fireStillBurning = true;
    steps = 0;
    fireReachedBottom = false;
    
    while (fireStillBurning) {
        // Determine which rows this process owns
        int i0, i1;
        distribute_grid(N, iproc, nproc, i0, i1);
        int localRows = i1 - i0;

        // Build the local updated slice
        // (reserve is fine, but resize is simpler to ensure data() is valid)
        vector<int> localNew(localRows * N, 0);

        // Update each cell in our slice
        for (int i = i0; i < i1; i++){
            for (int j = 0; j < N; j++){
                int idx = i * N + j;
                int current = globalGrid[idx];
                int newState = current;

                if (current == TREE) {
                    bool neighborBurning = false;
                    // Up
                    if (i > 0 && globalGrid[(i - 1) * N + j] == BURNING)
                        neighborBurning = true;
                    // Down
                    if (i < N - 1 && globalGrid[(i + 1) * N + j] == BURNING)
                        neighborBurning = true;
                    // Left
                    if (j > 0 && globalGrid[i * N + (j - 1)] == BURNING)
                        neighborBurning = true;
                    // Right
                    if (j < N - 1 && globalGrid[i * N + (j + 1)] == BURNING)
                        neighborBurning = true;

                    if (neighborBurning) {
                        newState = BURNING;
                    }
                }
                else if (current == BURNING) {
                    newState = DEAD;
                }

                // Store new state in local slice
                localNew[(i - i0) * N + j] = newState;
            }
        }

        // Gather the updated slices from all processes into newGlobalGrid
        vector<int> newGlobalGrid(N * N, 0);  // ALLOCATE ON ALL RANKS

        // Build recvCounts and displs on all ranks
        vector<int> recvCounts(nproc, 0), displs(nproc, 0);
        {
            int disp = 0;
            for (int p = 0; p < nproc; p++){
                int i0p, i1p;
                distribute_grid(N, p, nproc, i0p, i1p);
                int count = (i1p - i0p) * N;
                recvCounts[p] = count;
                displs[p] = disp;
                disp += count;
            }
        }

        // Perform the all-gatherv
        int localCount = localRows * N;
        MPI_Allgatherv(localNew.data(), localCount, MPI_INT,
                       newGlobalGrid.data(), recvCounts.data(), displs.data(), 
                       MPI_INT, MPI_COMM_WORLD);

        // Update global grid
        globalGrid = newGlobalGrid;

        // Check if any cell is still burning
        int localBurning = 0;
        for (int i = i0; i < i1; i++){
            for (int j = 0; j < N; j++){
                if (globalGrid[i * N + j] == BURNING) {
                    localBurning = 1;
                    break;
                }
            }
            if (localBurning) break;
        }
        int globalBurning = 0;
        // we use logical OR to see if any rank has burning
        MPI_Allreduce(&localBurning, &globalBurning, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        fireStillBurning = (globalBurning != 0);

        // Check if the fire reached the bottom row
        int localBottom = 0;
        if (i1 == N) { // only the rank that owns the bottom row checks
            for (int j = 0; j < N; j++){
                if (globalGrid[(N - 1) * N + j] == BURNING) {
                    localBottom = 1;
                    break;
                }
            }
        }
        int globalBottom = 0;
        MPI_Allreduce(&localBottom, &globalBottom, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (globalBottom)
            fireReachedBottom = true;

        steps++;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int iproc, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    // Expect at least 3 arguments: N, p, M
    if (argc < 4) {
        if (iproc == 0)
            cerr << "Usage: " << argv[0] 
                 << " N p M [input_filename] [output_filename]" << endl;
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    double p = atof(argv[2]);
    int M = atoi(argv[3]);
    string inputFile = "";
    string outputFile = "final_grid.txt";
    if (argc >= 5) inputFile = argv[4];
    if (argc >= 6) outputFile = argv[5];
    
    double t_start = MPI_Wtime();
    
    int totalSteps = 0;
    int runsFireReachedBottom = 0;
    
    // Perform M runs
    for (int run = 0; run < M; run++){
        vector<int> globalGrid;

        // If input file is provided, read from file on root
        if (!inputFile.empty()) {
            if (iproc == 0) {
                read_grid(inputFile, globalGrid, N);
            }
            // broadcast N so all ranks know grid size
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // resize on non-root
            if (iproc != 0) {
                globalGrid.resize(N * N, 0);
            }
            // broadcast the grid
            MPI_Bcast(globalGrid.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            // Generate random grid on root
            if (iproc == 0) {
                srand(time(NULL) + run); 
                generate_random_grid(N, p, globalGrid);

                // Ignite top row
                for (int j = 0; j < N; j++){
                    if (globalGrid[j] == TREE) {
                        globalGrid[j] = BURNING;
                    }
                }
            }
            else {
                // allocate on other ranks
                globalGrid.resize(N*N, 0);
            }
            // broadcast the grid
            MPI_Bcast(globalGrid.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Run the simulation
        int steps = 0;
        bool fireReachedBottom = false;
        simulate_fire(N, globalGrid, iproc, nproc, steps, fireReachedBottom);
        
        // Collect results
        totalSteps += steps;
        if (fireReachedBottom) {
            runsFireReachedBottom++;
        }
        
        // For the last run, root writes out final grid
        if (run == M - 1 && iproc == 0) {
            write_grid(outputFile, globalGrid, N);
        }
    }
    
    double t_finish = MPI_Wtime();
    double avgSteps = (double) totalSteps / M;
    
    // Root prints final summary
    if (iproc == 0) {
        cout << "Average number of steps: " << avgSteps << endl;
        cout << "Fraction of runs where fire reached bottom: " 
             << (double) runsFireReachedBottom / M << endl;
        cout << "Total simulation time: " << (t_finish - t_start) << " seconds" << endl;
    }
    
    MPI_Finalize();
    return 0;
}
