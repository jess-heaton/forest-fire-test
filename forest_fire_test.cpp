#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

enum State { EMPTY = 0, TREE = 1, BURNING = 2, DEAD = 3 };

vector<vector<int>> initialize_random_grid(int N, double p, mt19937& gen) {
    vector<vector<int>> grid(N, vector<int>(N, EMPTY));
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (dis(gen) < p) grid[i][j] = TREE;
        }
    }
    for (int j = 0; j < N; j++) {
        if (grid[0][j] == TREE) grid[0][j] = BURNING;
    }
    return grid;
}

vector<vector<int>> read_grid_from_file(const string& filename, int N) {
    vector<vector<int>> grid(N, vector<int>(N, EMPTY));
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file >> grid[i][j];
        }
    }
    file.close();
    for (int j = 0; j < N; j++) {
        if (grid[0][j] == TREE) grid[0][j] = BURNING;
    }
    return grid;
}

void print_grid(const vector<vector<int>>& grid, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << grid[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

bool simulate_step(vector<vector<int>>& grid, int N) {
    vector<vector<int>> new_grid = grid;
    bool fire_active = false;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] == BURNING) {
                new_grid[i][j] = DEAD;
                if (i > 0 && grid[i-1][j] == TREE) new_grid[i-1][j] = BURNING;
                if (i < N-1 && grid[i+1][j] == TREE) new_grid[i+1][j] = BURNING;
                if (j > 0 && grid[i][j-1] == TREE) new_grid[i][j-1] = BURNING;
                if (j < N-1 && grid[i][j+1] == TREE) new_grid[i][j+1] = BURNING;
            }
        }
    }
    grid = new_grid;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] == BURNING) fire_active = true;
        }
    }
    return fire_active;
}

pair<int, bool> run_simulation(vector<vector<int>>& grid, int N) {
    int steps = 0;
    bool fire_reached_bottom = false;
    bool fire_active = true;

    while (fire_active) {
        fire_active = simulate_step(grid, N);
        steps++;
        for (int j = 0; j < N; j++) {
            if (grid[N-1][j] == BURNING || grid[N-1][j] == DEAD) {
                fire_reached_bottom = true;
            }
        }
    }
    return {steps, fire_reached_bottom};
}

int main(int argc, char* argv[]) {
    int N = 10;
    double p = 0.6;
    int M = 5;
    bool use_file = false;
    string filename = "initial_grid.txt";

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) p = atof(argv[2]);
    if (argc > 3) M = atoi(argv[3]);
    if (argc > 4) {
        use_file = true;
        filename = argv[4];
    }

    random_device rd;
    mt19937 gen(rd());

    double total_time = 0;
    vector<int> all_steps(M);
    vector<bool> all_reached_bottom(M);

    for (int run = 0; run < M; run++) {
        vector<vector<int>> grid;
        if (use_file && run == 0) {
            grid = read_grid_from_file(filename, N);
        } else {
            grid = initialize_random_grid(N, p, gen);
        }

        auto start = chrono::high_resolution_clock::now();
        auto [steps, reached_bottom] = run_simulation(grid, N);
        auto end = chrono::high_resolution_clock::now();

        double time_taken = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
        total_time += time_taken;

        all_steps[run] = steps;
        all_reached_bottom[run] = reached_bottom;

        cout << "Run " << run + 1 << ": Steps = " << steps 
             << ", Reached bottom = " << (reached_bottom ? "Yes" : "No") 
             << ", Time = " << time_taken << "s" << endl;
    }

    double avg_steps = 0;
    int bottom_count = 0;
    for (int i = 0; i < M; i++) {
        avg_steps += all_steps[i];
        if (all_reached_bottom[i]) bottom_count++;
    }
    avg_steps /= M;
    double avg_time = total_time / M;

    cout << "\nSummary for N=" << N << ", p=" << p << ", M=" << M << ":\n";
    cout << "Average steps: " << avg_steps << endl;
    cout << "Fraction reaching bottom: " << (double)bottom_count / M << endl;
    cout << "Average time per run: " << avg_time << "s" << endl;

    return 0;
}