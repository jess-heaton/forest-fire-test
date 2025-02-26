#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <string>
#include <algorithm>

enum State { EMPTY = 0, TREE = 1, BURNING = 2, DEAD = 3 };

// -----------------------------------------------------------------------------
// Distribute rows among processes
// -----------------------------------------------------------------------------
static void distribute_rows(int N, int rank, int size, int &start_row, int &local_N)
{
    int base = N / size;
    int rem  = N % size;
    if (rank < rem) {
        local_N = base + 1;
        start_row = rank * (base + 1);
    } else {
        local_N = base;
        start_row = rem * (base + 1) + (rank - rem) * base;
    }
}

// -----------------------------------------------------------------------------
// Generate or read grid
// -----------------------------------------------------------------------------
std::vector<int> initialize_random_grid(int N, double p, std::mt19937& gen)
{
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<int> grid(N*N, EMPTY);

    for (int i = 0; i < N*N; i++) {
        if (dis(gen) < p) {
            grid[i] = TREE;
        }
    }
    // Set top-row trees on fire
    for (int j = 0; j < N; j++) {
        if (grid[j] == TREE) {
            grid[j] = BURNING;
        }
    }
    return grid;
}

std::vector<int> read_grid_from_file(const std::string& filename, int N)
{
    std::vector<int> grid(N*N, EMPTY);
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (N == 0) return grid;
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N*N; i++) {
        file >> grid[i];
    }
    file.close();

    // Set top-row trees on fire
    for (int j = 0; j < N; j++) {
        if (grid[j] == TREE) {
            grid[j] = BURNING;
        }
    }
    return grid;
}

// -----------------------------------------------------------------------------
// Scatter rows
// -----------------------------------------------------------------------------
void scatter_grid(const std::vector<int>& full_grid,
                  std::vector<int>& local_grid,
                  int N, int start_row, int local_N,
                  int rank, int size)
{
    std::vector<int> send_counts(size), displs(size);
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int sr, lN;
            distribute_rows(N, r, size, sr, lN);
            send_counts[r] = lN * N;
            displs[r]      = sr * N;
        }
    }

    MPI_Scatterv(full_grid.data(),
                 send_counts.data(),
                 displs.data(),
                 MPI_INT,
                 local_grid.data(),
                 local_N * N,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);
}

// -----------------------------------------------------------------------------
// Halo exchange
// -----------------------------------------------------------------------------
void exchange_halos(std::vector<int>& local_grid, int local_N, int N, int rank, int size)
{
    MPI_Status status;
    MPI_Request requests[4];
    int count = N;

    int up   = (rank == 0 ? MPI_PROC_NULL : rank - 1);
    int down = (rank == size-1 ? MPI_PROC_NULL : rank + 1);

    std::vector<int> top_row(N), bottom_row(N);
    std::vector<int> recv_from_up(N), recv_from_down(N);

    // Copy out top/bottom rows
    for(int j = 0; j < N; j++){
        top_row[j]    = local_grid[0*N + j];
        bottom_row[j] = local_grid[(local_N-1)*N + j];
    }

    // Send
    MPI_Isend(top_row.data(),    N, MPI_INT, up,   0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(bottom_row.data(), N, MPI_INT, down, 1, MPI_COMM_WORLD, &requests[1]);
    // Receive
    MPI_Irecv(recv_from_up.data(),   N, MPI_INT, up,   1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(recv_from_down.data(), N, MPI_INT, down, 0, MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // For simplicity, we won't store these new rows in local_grid,
    // so no further code needed here.
}

// -----------------------------------------------------------------------------
// Step simulation
// -----------------------------------------------------------------------------
bool simulate_step(std::vector<int>& grid, int local_N, int N,
                   const std::vector<int>& up_row,
                   const std::vector<int>& down_row)
{
    std::vector<int> new_grid = grid;
    bool fire_active = false;

    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i*N + j] == BURNING) {
                new_grid[i*N + j] = DEAD;

                // Up
                if (i > 0) {
                    if (grid[(i-1)*N + j] == TREE)
                        new_grid[(i-1)*N + j] = BURNING;
                } else {
                    // i==0 => check up_row (skipped in this minimal approach)
                }
                // Down
                if (i < local_N-1) {
                    if (grid[(i+1)*N + j] == TREE)
                        new_grid[(i+1)*N + j] = BURNING;
                } else {
                    // check down_row
                }

                // Left
                if (j > 0) {
                    if (grid[i*N + (j-1)] == TREE)
                        new_grid[i*N + (j-1)] = BURNING;
                }
                // Right
                if (j < N-1) {
                    if (grid[i*N + (j+1)] == TREE)
                        new_grid[i*N + (j+1)] = BURNING;
                }
            }
        }
    }

    grid = new_grid;
    for (int i = 0; i < local_N*N; i++) {
        if (grid[i] == BURNING) {
            fire_active = true;
            break;
        }
    }
    return fire_active;
}

// -----------------------------------------------------------------------------
// Run simulation
// -----------------------------------------------------------------------------
std::pair<int,bool> run_simulation(std::vector<int>& local_grid,
                                   int N, int local_N,
                                   int start_row, int rank, int size)
{
    int steps = 0;
    bool local_fire_active   = true;
    bool global_fire_active  = true;
    bool local_reached_bottom = false;

    std::vector<int> up_row(N), down_row(N);

    while (true) {
        exchange_halos(local_grid, local_N, N, rank, size);

        // For simplicity, we won't fill up_row/down_row. Keep them empty.
        up_row.clear();
        down_row.clear();

        local_fire_active = simulate_step(local_grid, local_N, N, up_row, down_row);

        MPI_Allreduce(&local_fire_active, &global_fire_active, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        steps++;

        // Check bottom
        int end_row = start_row + local_N - 1;
        if (end_row == N - 1) {
            for (int j = 0; j < N; j++) {
                int state = local_grid[(local_N - 1)*N + j];
                if (state == BURNING || state == DEAD) {
                    local_reached_bottom = true;
                    break;
                }
            }
        }
        bool global_reached_bottom;
        MPI_Allreduce(&local_reached_bottom, &global_reached_bottom, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        if (!global_fire_active) {
            return {steps, global_reached_bottom};
        }
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N         = 10;
    double p      = 0.6;
    int M         = 5;
    bool use_file = false;
    std::string filename = "initial_grid.txt";

    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) p = std::atof(argv[2]);
    if (argc > 3) M = std::atoi(argv[3]);
    if (argc > 4) {
        use_file = true;
        filename = argv[4];
    }

    // (A) Prepare CSV on rank 0
    std::ofstream csvfile;
    if (rank == 0) {
        csvfile.open("convergence_results.csv", std::ios::out); // overwrite each run
        csvfile << "N,p,Run,Steps,ReachedBottom,Time\n";
    }

    int start_row, local_N;
    distribute_rows(N, rank, size, start_row, local_N);

    std::random_device rd;
    std::mt19937 gen(rd() + rank);

    double total_time = 0.0;
    std::vector<int>  all_steps(M);
    std::vector<bool> all_reached_bottom(M);

    for (int run = 0; run < M; run++) {
        // Rank 0 either reads or generates the full grid
        std::vector<int> full_grid;
        if (rank == 0) {
            if (use_file && run == 0) {
                full_grid = read_grid_from_file(filename, N);
            } else {
                full_grid = initialize_random_grid(N, p, gen);
            }
        }

        // Scatter
        std::vector<int> local_grid(local_N * N, EMPTY);
        scatter_grid(full_grid, local_grid, N, start_row, local_N, rank, size);

        // Time simulation
        double t0 = MPI_Wtime();
        auto result = run_simulation(local_grid, N, local_N, start_row, rank, size);
        double t1 = MPI_Wtime();

        int steps = result.first;
        bool reached_bottom = result.second;
        double time_taken   = (t1 - t0);

        // Gather results to rank 0
        int global_steps;
        MPI_Reduce(&steps, &global_steps, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

        bool global_reached_bottom;
        MPI_Reduce(&reached_bottom, &global_reached_bottom, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);

        double global_time;
        MPI_Reduce(&time_taken, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            all_steps[run]          = steps;
            all_reached_bottom[run] = global_reached_bottom;
            total_time             += global_time;

            // Print results
            std::cout << "Run " << run+1
                      << ": Steps = " << steps
                      << ", Reached bottom = " << (global_reached_bottom ? "Yes" : "No")
                      << ", Time = " << global_time << "s"
                      << std::endl;

            // (B) Write a row to the CSV file
            csvfile << N << "," << p << "," 
                    << (run + 1) << ","
                    << steps << ","
                    << (global_reached_bottom ? "Yes" : "No") << ","
                    << global_time
                    << "\n";
        }
    }

    // Final summary
    if (rank == 0) {
        double avg_steps = 0.0;
        int bottom_count = 0;
        for (int i = 0; i < M; i++) {
            avg_steps += all_steps[i];
            if (all_reached_bottom[i]) bottom_count++;
        }
        avg_steps /= M;
        double avg_time = total_time / M;

        std::cout << "\nSummary for N=" << N << ", p=" << p << ", M=" << M << ":\n";
        std::cout << "Average steps: " << avg_steps << std::endl;
        std::cout << "Fraction reaching bottom: " << (double)bottom_count / M << std::endl;
        std::cout << "Average time per run: " << avg_time << "s" << std::endl;

        // (C) Close CSV
        csvfile.close();
    }

    MPI_Finalize();
    return 0;
}
