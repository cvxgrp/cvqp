#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath> // Include the math library
#include <tuple>
#include "sum_largest_proj.h"

void form_delta(int untied, int tied, int k, int len_z, std::pair<double,double>& p) {
    int u = untied;
    int t = tied;
    int n = k - u;

    int len = (u + t < len_z) ? (u + t + 1): len_z;

    double untied_val = (n > 0) ? (static_cast<double>(t) / n) : 1.0;

    double val_1 = (untied > 0) ? untied_val : 1.0;
    double val_2 = (tied > 0) ? 1.0 : 0.0;

    double normalization;
    if (k - u > 0) {
        double a = u * t;
        double b = k - u;
        double t1 = a / b;
        double t2 = k - u;
        normalization = t1 + t2;
    } else {
        normalization = k;
    }

    val_1 /= normalization;
    val_2 /= normalization;

    // form a pair of val_1 and val_2
    p.first = val_1;
    p.second = val_2;
}

std::tuple<int,int, bool> sum_largest_proj(double* z, int n, int k, double alpha, int untied, int tied, int cutoff,  bool debug) {  
    
    // std::cout << "debug: " << debug << std::endl;
    
    // compute the sum of the k largest elements in z
    double val = 0.0;
    for (int i = 0; i < k; i++) {
        val += z[i];
    }

    // int untied = k;
    // int tied = 0;
    int final_untied_count;
    int final_tied_count;
    int iters = 0;
    const double TOL = 1e-9;

    double s1;
    double s2;
    double s;
    bool valid_s1;
    bool valid_s2;
    double extra = val - alpha;
    double untied_val;
    double tied_val;
    double end_val;
    double penultimate_val;
    double d;
    std::pair<double, double> p;

    double untied_decrease = 0.0;
    
    double tied_final = z[k];
    double last_untied_val = z[k - 1];
    double post_tied_val = z[k];

    double lim_inf = std::numeric_limits<double>::infinity();

    int MAX_ITERS = n;
    while ((val > alpha + TOL) && (iters < MAX_ITERS) && (tied + untied <= cutoff))
    {
        form_delta(untied, tied, k, n, p);
        extra = val - alpha;
        if (debug) {
            std::cout << "current iterate: ";
         
            for (int i = 0; i < untied; i++) {
                std::cout << z[i] - untied_decrease << " ";
            }
            for (int i = untied; i < untied + tied; i++) {
                std::cout << tied_final << " ";
            }
            for (int i = untied + tied; i < n; i++) {
                std::cout << z[i] << " ";
            }
            std::cout << std::endl;

            // print val
            std::cout << "val: " << val << std::endl;

            // print k, untied, and tied
            std::cout << "k: " << k << std::endl;
            std::cout << "untied: " << untied << std::endl;
            std::cout << "tied: " << tied << std::endl;
        }

        untied_val = p.first;
        tied_val = p.second;

        valid_s1 = untied > 0 ? true : false;

        // if (debug) {
        //     std::cout << "untied val: " << untied_val << std::endl;
        //     std::cout << "tied val: " << tied_val << std::endl;
        // }

        if (valid_s1) {
            // s1 = (z[untied] - z[untied - 1]) / (delta[untied] - delta[untied - 1]);
            // z[untied] = tied_final;
            // z[untied - 1] = last_untied_val;
            s1 = (tied_final - last_untied_val) / (tied_val - untied_val);

            // if (debug) {
            //     std::cout << "z[untied]: " << z[untied] << std::endl;
            //     std::cout << "tied_final: " << tied_final << std::endl;
            //     std::cout << "z[untied - 1]: " << z[untied - 1] << std::endl;
            //     std::cout << "last_untied_val: " << last_untied_val << std::endl;
            // }

        } else {
            s1 = lim_inf;
        }
        valid_s2 = untied + tied < n ? true : false;
        if (valid_s2) {
            // s2 = (z[untied + tied] - z[untied + tied - 1]) / (delta[untied + tied] - delta[untied + tied - 1]);

            if (tied == 0) {
                penultimate_val = untied_val;
            } else {
                penultimate_val = tied_val;
            }
            // s2 = (z[untied + tied] - z[untied + tied - 1]) / (0.0 - penultimate_val);

            double v = (tied == 0) ? last_untied_val : tied_final;
            s2 = (post_tied_val - v) / (0.0 - penultimate_val);

            // if (debug) {
            //     std::cout << "z[untied + tied]: " << z[untied + tied] << std::endl;
            //     std::cout << "post_tied_val: " << post_tied_val << std::endl;
            //     std::cout << "z[untied + tied - 1]: " << z[untied + tied - 1] << std::endl;
            //     std::cout << "final: " << v << std::endl;
            //     std::cout << "penultimate_val: " << penultimate_val << std::endl;
            // }

        } else {
            s2 = lim_inf;
        }

        s = std::min(s1, s2);
        s = std::min(s, extra);

        if (debug) {
            std::cout << "s1: " << s1 << std::endl;
            std::cout << "s2: " << s2 << std::endl;
            std::cout << "s: " << s << std::endl;
        }

        val -= s * untied * untied_val;
        val -= s * static_cast<double>(k - untied) * tied_val;

        if (tied > 0 ){
            tied_final -= s * tied_val;
        }
        untied_decrease += s * untied_val;

        final_untied_count = untied;
        final_tied_count = tied;

        untied = (s == s1) ? std::max(untied - 1, 0) : untied;
        if (tied == 0) {
            tied = 1;
        }
        tied = std::min(tied + 1, static_cast<int>(n));

        if (untied > 0) {
            last_untied_val = z[untied - 1] - untied_decrease;
        }
        if (untied + tied < n) {
            post_tied_val = z[untied + tied];
        }
      
        iters++;
        // std::cout << "z0: " << z[0] << std::endl;
    }    

    // std::cout << "z0, again: " << z[0] << std::endl;
    for (int i = 0; i < final_untied_count; i++) {
        z[i] -= untied_decrease; 
    }
    for (int i = final_untied_count; i < final_untied_count + final_tied_count; i++) {
        z[i] = tied_final;
    }

    // std::cout << "dec: " << untied_decrease << std::endl;
    // std::cout << "tie: " << tied_final << std::endl;

    // std::cout << "final iterate: ";
    // for (int i = 0; i < n; i++) {
    //     std::cout << z[i] << " ";
    // }
    // std::cout << std::endl;


    // std::cout << "final sum: " << val << std::endl;
    // double test_sum = 0.0;
    // for (int i = 0; i < n; i++) {
    //     test_sum += z[i];
    // }
    // std::cout << "test sum: " << test_sum << std::endl;


    bool complete = (val <= alpha + TOL) ? true : false;

    // return final_tied_count + final_untied_count;
    // make a pair of (final_tied_count + final_untied_count, complete)
    // std::pair<int, bool> r = std::make_pair(final_tied_count + final_untied_count, complete);
    std::tuple<int,int, bool> r = std::make_tuple(final_untied_count, final_tied_count, complete);
    return r;
}

// you can ignore this, it was just for profiling. I time the c++ and python in
// a separate py script
int main() {
    const int trials = 1;
    std::vector<int> iterCounts(trials);
    std::vector<double> times(trials);

    // Seed for random number generation
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < trials; i++) {
        
        // Generate random input
        int n = 100000;
        int k = 5000;
        // double alpha = 0.5 * n;
        std::vector<double> z(n);

        // n = 100_000
        // k = int(.05*n)

        // z = np.random.uniform(0, 20, n)
        // alpha = np.random.uniform(0, 15)

        double alpha = static_cast<double>(std::rand()) / RAND_MAX * 15.0;

        for (int j = 0; j < n; j++) {
            z[j] = static_cast<double>(std::rand()) / RAND_MAX * 20.0;
        }
        
        // sort z in descending order
        std::sort(z.begin(), z.end(), std::greater<double>());

        clock_t start = std::clock();

        // Call your sum_largest_proj function

        // pointer to z
        double* z_ptr = &z[0];
        
        std::tuple<int, int, bool> r = sum_largest_proj(z_ptr, n, k, alpha, k, 0, n, false);
    
        int iters = std::get<0>(r) + std::get<1>(r);
        bool complete = std::get<2>(r);

        clock_t end = std::clock();
        double elapsedSeconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        iterCounts[i] = iters;
        times[i] = elapsedSeconds;
    }

    // Calculate mean and standard deviation of times
    double sum = 0.0;
    for (int i = 0; i < trials; i++) {
        sum += times[i];
    }
    double mean = sum / trials;

    double sumOfSquares = 0.0;
    for (int i = 0; i < trials; i++) {
        double diff = times[i] - mean;
        sumOfSquares += diff * diff;
    }
    double stdDeviation = std::sqrt(sumOfSquares / trials);

    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard Deviation: " << stdDeviation << std::endl;

    return 0;
}