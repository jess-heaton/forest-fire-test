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

// This function generates a random grid of size N x N.
// Each cell gets a TREE with probability p, otherwise it stays EMPTY.
void generate_random_grid(int N, double p, vector<int>& grid) {
    grid.resize(N * N, EMPTY);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            double r = (double) rand() / RAND_MAX;
            if (r < p)
                grid[i * N + j] = TREE;
            else
                grid[i * N + j] = EMPTY;
        }
    }
}

// Read a grid from a file.
// The file format is assumed to have the grid size N on the first line
// and then N rows with N numbers per row.
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

// Divide the grid rows among MPI tasks.
// Each process gets a slice of rows defined by [i0, i1).
void distribute_grid(int N, int iproc, int nproc, int & i0, int & i1) {
    int rows = N / nproc;
    i0 = iproc * rows;
    if (iproc == nproc - 1)
        i1 = N; // last process takes any extra rows
    else
        i1 = i0 + rows;
}

// This function performs the simulation of the forest fire.
// It updates the grid until no more burning trees are present.
// It outputs the number of steps and whether the fire reached the bottom row.
void simulate_fire(int N, vector<int>& globalGrid, int iproc, int nproc, int& steps, bool & fireReachedBottom) {
    bool fireStillBurning = true;
    steps = 0;
    fireReachedBottom = false;
    
    // Loop until there are no more burning trees.
    while (fireStillBurning) {
        // Determine which rows this process is responsible for.
        int i0, i1;
        distribute_grid(N, iproc, nproc, i0, i1);
        int localRows = i1 - i0;
        
        // Create a vector to hold the updated values for our slice.
        vector<int> localNew;
        localNew.reserve(localRows * N);
        
        // Loop over our portion of rows.
        for (int i = i0; i < i1; i++){
            for (int j = 0; j < N; j++){
                int idx = i * N + j;
                int current = globalGrid[idx];
                int newState = current;
                // If the cell is a TREE, check if any neighbour is burning.
                if (current == TREE) {
                    bool neighborBurning = false;
                    // Up neighbor
                    if (i > 0 && globalGrid[(i - 1) * N + j] == BURNING)
                        neighborBurning = true;
                    // Down neighbor
                    if (i < N - 1 && globalGrid[(i + 1) * N + j] == BURNING)
                        neighborBurning = true;
                    // Left neighbor
                    if (j > 0 && globalGrid[i * N + (j - 1)] == BURNING)
                        neighborBurning = true;
                    // Right neighbor
                    if (j < N - 1 && globalGrid[i * N + (j + 1)] == BURNING)
                        neighborBurning = true;
                    
                    if (neighborBurning)
                        newState = BURNING;
                }
                // If the cell is burning, it becomes dead.
                else if (current == BURNING) {
                    newState = DEAD;
                }
                // Otherwise, the state stays the same.
                localNew.push_back(newState);
            }
        }
        
        // Now we need to combine our updated slice with the other processes.
        // We use MPI_Allgatherv to collect each process’s localNew into a new global grid.
        vector<int> newGlobalGrid;
        if (iproc == 0)
            newGlobalGrid.resize(N * N, 0);
            
        int localCount = localRows * N;
        // Prepare arrays for the counts and displacements (only used on the root).
        vector<int> recvCounts, displs;
        if (iproc == 0) {
            recvCounts.resize(nproc);
            displs.resize(nproc);
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
        
        MPI_Allgatherv(localNew.data(), localCount, MPI_INT,
                       newGlobalGrid.data(), recvCounts.data(), displs.data(), MPI_INT,
                       MPI_COMM_WORLD);
        // Update the global grid with our new values.
        globalGrid = newGlobalGrid;
        
        // Now check if there are still any burning trees.
        int localBurning = 0;
        for (int i = i0; i < i1; i++){
            for (int j = 0; j < N; j++){
                if (globalGrid[i * N + j] == BURNING) {
                    localBurning = 1;
                    break;
                }
            }
            if (localBurning)
                break;
        }
        int globalBurning = 0;
        MPI_Allreduce(&localBurning, &globalBurning, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        fireStillBurning = (globalBurning != 0);
        
        // Check if the fire reached the bottom row.
        int localBottom = 0;
        if (i1 == N) { // only the process that owns the bottom row does this
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
        
        steps++;  // increment the time step
    }
}

//
// Main function
//
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int iproc, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    // We expect the following command line arguments:
    // argv[1]: Grid size N (e.g. 100)
    // argv[2]: Tree probability p (e.g. 0.6)
    // argv[3]: Number of independent runs M (e.g. 50)
    // argv[4] (optional): Input filename (if you want to read an initial grid)
    // argv[5] (optional): Output filename (for the final grid of the last run)
    if (argc < 4) {
        if (iproc == 0)
            cout << "Usage: " << argv[0] << " N p M [input_filename] [output_filename]" << endl;
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    double p = atof(argv[2]);
    int M = atoi(argv[3]);
    string inputFile = "";
    string outputFile = "final_grid.txt";
    if (argc >= 5)
        inputFile = argv[4];
    if (argc >= 6)
        outputFile = argv[5];
    
    // Timing start.
    double t_start = MPI_Wtime();
    
    int totalSteps = 0;
    int runsFireReachedBottom = 0;
    
    // Loop over M independent runs.
    for (int run = 0; run < M; run++){
        vector<int> globalGrid;
        
        // If an input file is provided, read the grid from the file.
        // Otherwise, generate a random grid.
        if (inputFile != "") {
            if (iproc == 0) {
                read_grid(inputFile, globalGrid, N);
            }
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (iproc != 0)
                globalGrid.resize(N * N, 0);
            MPI_Bcast(globalGrid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            // Generate a random grid on the root process.
            if (iproc == 0) {
                srand(time(NULL) + run); // different seed for each run
                generate_random_grid(N, p, globalGrid);
            }
            else {
                globalGrid.resize(N * N, 0);
            }
            // Broadcast the generated grid to all tasks.
            MPI_Bcast(globalGrid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Ignite the fire in the top row: if a cell has a TREE, set it to BURNING.
            if (iproc == 0) {
                for (int j = 0; j < N; j++){
                    if (globalGrid[j] == TREE)
                        globalGrid[j] = BURNING;
                }
            }
            MPI_Bcast(globalGrid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        int steps = 0;
        bool fireReachedBottom = false;
        simulate_fire(N, globalGrid, iproc, nproc, steps, fireReachedBottom);
        
        totalSteps += steps;
        if (fireReachedBottom)
            runsFireReachedBottom++;
        
        // For the last run, the root writes out the final grid.
        if (run == M - 1 && iproc == 0) {
            write_grid(outputFile, globalGrid, N);
        }
    }
    
    double t_finish = MPI_Wtime();
    double avgSteps = (double) totalSteps / M;
    
    // Only the root prints out the results.
    if (iproc == 0) {
        cout << "Average number of steps: " << avgSteps << endl;
        cout << "Fraction of runs where fire reached bottom: " 
             << (double) runsFireReachedBottom / M << endl;
        cout << "Total simulation time: " << t_finish - t_start << " seconds" << endl;
    }
    
    MPI_Finalize();
    return 0;
}
