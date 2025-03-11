
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <string>

using namespace std;

// cell state definitions
enum CellState { EMPTY = 0, TREE = 1, BURNING = 2, DEAD = 3 };

//---------------------------------------------------------------------
// Read grid from file (only used in file mode, on rank 0)
// File format: first line contains grid size N, then N lines with N integers each.
// After reading the grid, the code sets any tree in the top row to burning.
vector<int> read_grid(const string& filename, int &N) {
    ifstream fin(filename.c_str());
    if (!fin) {
        cerr << "Error opening file " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fin >> N;
    vector<int> grid(N * N, EMPTY);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            fin >> grid[i * N + j];
        }
    }
    fin.close();
    // set top row trees to burning
    for (int j = 0; j < N; j++){
        if (grid[j] == TREE)
            grid[j] = BURNING;
    }
    return grid;
}

//---------------------------------------------------------------------
// Generate a random grid of size N. Each cell gets a tree (state TREE)
// with probability p; otherwise it is empty. Then, set top row trees to burning.
vector<int> generate_grid(int N, double p) {
    vector<int> grid(N * N, EMPTY);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            double r = (double) rand() / RAND_MAX;
            grid[i * N + j] = (r < p) ? TREE : EMPTY;
        }
    }
    // set top row trees to burning
    for (int j = 0; j < N; j++){
        if (grid[j] == TREE)
            grid[j] = BURNING;
    }
    return grid;
}

//---------------------------------------------------------------------
// Divide the grid rows among MPI tasks (row decomposition)
// Computes the global start and end indices (i0 and i1) for each process.
void distribute_grid(int N, int iproc, int nproc, int &i0, int &i1) {
    int rows_per_proc = N / nproc;
    i0 = iproc * rows_per_proc;
    if (iproc == nproc - 1)
        i1 = N;
    else
        i1 = i0 + rows_per_proc;
}

//---------------------------------------------------------------------
// Update the local grid for one time step.
// The local grid is stored with halo rows. The interior rows are indices 1 to local_rows.
// For each cell in an interior row, the update rules are:
//   - If a cell is TREE and any (up, down, left, right) neighbor is BURNING in the old grid,
//     it becomes BURNING.
//   - If a cell is BURNING, it becomes DEAD.
//   - Otherwise, the cell remains unchanged.
// Returns true if at least one cell in the updated (new) grid becomes BURNING.
bool update_local_grid(const vector<int>& local_old, vector<int>& local_new, int local_rows, int N) {
    bool local_burning = false;
    // Loop over interior rows (1...local_rows) and all columns
    for (int i = 1; i <= local_rows; i++){
        for (int j = 0; j < N; j++){
            int idx = i * N + j;
            int current = local_old[idx];
            if (current == TREE) {
                bool neighbor_burning = false;
                // Up neighbor (row i-1)
                if (local_old[(i - 1) * N + j] == BURNING)
                    neighbor_burning = true;
                // Down neighbor (row i+1)
                if (local_old[(i + 1) * N + j] == BURNING)
                    neighbor_burning = true;
                // Left neighbor
                if (j > 0 && local_old[i * N + (j - 1)] == BURNING)
                    neighbor_burning = true;
                // Right neighbor
                if (j < N - 1 && local_old[i * N + (j + 1)] == BURNING)
                    neighbor_burning = true;
                if (neighbor_burning) {
                    local_new[idx] = BURNING;
                    local_burning = true;
                } else {
                    local_new[idx] = TREE;
                }
            } else if (current == BURNING) {
                local_new[idx] = DEAD;
            } else {
                local_new[idx] = current;
            }
        }
    }
    return local_burning;
}

//---------------------------------------------------------------------
// Exchange halo rows with neighbouring processes.
// The local grid is stored as a 1D vector representing a 2D array with (local_rows+2) rows and N columns.
// Row 0 and row (local_rows+1) are the halo regions.
// This function uses MPI_Sendrecv to exchange the top and bottom halos.
void exchange_halos(vector<int>& local_grid, int local_rows, int N, int iproc, int nproc, MPI_Comm comm) {
    // Exchange with the upper neighbor (if exists)
    if (iproc > 0) {
        MPI_Sendrecv(&local_grid[1 * N], N, MPI_INT, iproc - 1, 0,
                     &local_grid[0], N, MPI_INT, iproc - 1, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    // Exchange with the lower neighbor (if exists)
    if (iproc < nproc - 1) {
        MPI_Sendrecv(&local_grid[local_rows * N], N, MPI_INT, iproc + 1, 1,
                     &local_grid[(local_rows + 1) * N], N, MPI_INT, iproc + 1, 1,
                     comm, MPI_STATUS_IGNORE);
    }
}

//---------------------------------------------------------------------
// Main function
// Supports two modes:
//   "random" mode: generate a random grid with given N and probability p
//   "file" mode: read the grid from a file
// In both cases, the simulation is repeated M times and the averaged results are output.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int nproc, iproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    // Process command-line arguments.
    // For random mode:  ./forest_fire random N p M output_filename
    // For file mode:    ./forest_fire file input_filename M output_filename
    bool random_mode;
    int M_runs;
    string output_filename;
    int N;       // grid size
    double p;  // probability for tree
    vector<int> full_grid;  // complete grid (only valid on rank 0)

    if (argc < 2) {
        if (iproc == 0)
            cout << "Usage:\n"
                 << "  For random grid: mpirun -np <nproc> ./forest_fire random N p M output_filename\n"
                 << "  For file input:  mpirun -np <nproc> ./forest_fire file input_filename M output_filename\n";
        MPI_Finalize();
        return 1;
    }
    
    string mode = argv[1];
    if (mode == "random") {
        if (argc != 6) {
            if (iproc == 0)
                cout << "Usage: mpirun -np <nproc> ./forest_fire random N p M output_filename" << endl;
            MPI_Finalize();
            return 1;
        }
        random_mode = true;
        N = atoi(argv[2]);
        p = atof(argv[3]);
        M_runs = atoi(argv[4]);
        output_filename = argv[5];
    } else if (mode == "file") {
        if (argc != 5) {
            if (iproc == 0)
                cout << "Usage: mpirun -np <nproc> ./forest_fire file input_filename M output_filename" << endl;
            MPI_Finalize();
            return 1;
        }
        random_mode = false;
        string input_filename = argv[2];
        M_runs = atoi(argv[3]);
        output_filename = argv[4];
        if (iproc == 0) {
            full_grid = read_grid(input_filename, N);
        }
    } else {
        if (iproc == 0)
            cout << "Unknown mode: " << mode << endl;
        MPI_Finalize();
        return 1;
    }
    
    // Broadcast grid size N to all processes.
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Variables to accumulate the results over runs.
    double total_steps = 0.0;
    int total_bottom = 0;
    double total_sim_time = 0.0;

    // Determine which global rows this process will handle.
    int i0, i1;
    distribute_grid(N, iproc, nproc, i0, i1);
    int local_rows = i1 - i0;  // number of rows in the local domain

    // For each independent run:
    for (int run = 0; run < M_runs; run++) {
        // On rank 0: generate (or already have read) the full grid.
        if (iproc == 0) {
            if (random_mode) {
                srand(time(NULL) + run); // different seed per run
                full_grid = generate_grid(N, p);
            }
            // In file mode, we use the same grid for all runs.
        }
        // Broadcast the full grid to all processes.
        if (nproc > 1) {
            if (iproc != 0)
                full_grid.resize(N * N);
            MPI_Bcast(full_grid.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Allocate local grids (with halo rows). Each local array has (local_rows+2) rows.
        vector<int> local_old((local_rows + 2) * N, 0);
        vector<int> local_new((local_rows + 2) * N, 0);
        
        // Copy the portion of the full grid corresponding to global rows [i0, i1)
        // into local_old rows 1 ... local_rows.
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < N; j++) {
                local_old[(i + 1) * N + j] = full_grid[(i0 + i) * N + j];
            }
        }
        // (The halo rows will be filled via MPI exchange.)
        
        bool bottom_reached = false; // flag to indicate if fire reached the bottom
        int steps = 0;              // count simulation steps

        double sim_start = MPI_Wtime();
        // Simulation loop: run until no new cells become burning.
        while (true) {
            // Exchange halo rows with neighboring MPI tasks.
            exchange_halos(local_old, local_rows, N, iproc, nproc, MPI_COMM_WORLD);
            
            // Update interior cells using the forest fire rules.
            // Only cells that change from TREE to BURNING will be counted.
            bool local_new_burning = update_local_grid(local_old, local_new, local_rows, N);
            
            // Check whether the fire reached the bottom.
            // (Only the process that owns the global bottom row needs to do this.)
            if (i1 == N) {
                for (int j = 0; j < N; j++) {
                    int state = local_new[local_rows * N + j]; // last interior row
                    if (state == BURNING || state == DEAD) {
                        bottom_reached = true;
                        break;
                    }
                }
            }
            
            // Check globally if any new burning cell exists.
            int local_flag = local_new_burning ? 1 : 0;
            int global_flag;
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (!global_flag)
                break;
            
            // Swap the new grid into the old grid for the next iteration.
            local_old.swap(local_new);
            steps++;
        }
        double sim_end = MPI_Wtime();
        double sim_time = sim_end - sim_start;
        
        // Determine (across all tasks) if the fire reached the bottom.
        int local_bottom = bottom_reached ? 1 : 0;
        int global_bottom;
        MPI_Allreduce(&local_bottom, &global_bottom, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        // Let rank 0 accumulate the results.
        if (iproc == 0) {
            total_steps += steps;
            total_sim_time += sim_time;
            total_bottom += global_bottom;
        }
    } // end run loop

    // Only rank 0 outputs the average results.
    if (iproc == 0) {
        double avg_steps = total_steps / M_runs;
        double avg_sim_time = total_sim_time / M_runs;
        double bottom_fraction = static_cast<double>(total_bottom) / M_runs;
        
        // Write the results to the output file.
        ofstream fout(output_filename.c_str());
        if (!fout) {
            cout << "Error opening output file " << output_filename << endl;
        } else {
            fout << "Average steps: " << avg_steps << "\n";
            fout << "Fraction of runs where fire reached bottom: " << bottom_fraction << "\n";
            fout << "Average simulation time (s): " << avg_sim_time << "\n";
            fout.close();
        }
        // Also print the results to standard output.
        cout << "Average steps: " << avg_steps << endl;
        cout << "Fraction of runs where fire reached bottom: " << bottom_fraction << endl;
        cout << "Average simulation time (s): " << avg_sim_time << endl;
    }
    
    MPI_Finalize();
    return 0;
}
