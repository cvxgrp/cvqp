#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include "proj_sum_largest.h"

void form_delta(int untied, int tied, int k, int len_z, std::pair<double,double>& p) {
    int u = untied;
    int t = tied;
    int n = k - u;

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

double compute_initial_sum(double* z, int k) {
    double val = 0.0;
    for (int i = 0; i < k; i++) {
        val += z[i];
    }
    return val;
}

void print_debug_info(double* z, int n, int untied, int tied, double untied_decrease, double tied_final, double val, int k) {
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
    
    std::cout << "val: " << val << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "untied: " << untied << std::endl;
    std::cout << "tied: " << tied << std::endl;
}

void compute_step_sizes(int untied, int tied, int n, double untied_val, double tied_val, 
                       double tied_final, double last_untied_val, double post_tied_val,
                       double& s1, double& s2) {
    double lim_inf = std::numeric_limits<double>::infinity();
    
    // Compute s1
    bool valid_s1 = untied > 0;
    if (valid_s1) {
        s1 = (tied_final - last_untied_val) / (tied_val - untied_val);
    } else {
        s1 = lim_inf;
    }
    
    // Compute s2
    bool valid_s2 = untied + tied < n;
    if (valid_s2) {
        double penultimate_val;
        if (tied == 0) {
            penultimate_val = untied_val;
        } else {
            penultimate_val = tied_val;
        }
        
        double v = (tied == 0) ? last_untied_val : tied_final;
        s2 = (post_tied_val - v) / (0.0 - penultimate_val);
    } else {
        s2 = lim_inf;
    }
}

void apply_final_projection(double* z, int final_untied_count, int final_tied_count, 
                           double untied_decrease, double tied_final) {
    for (int i = 0; i < final_untied_count; i++) {
        z[i] -= untied_decrease; 
    }
    for (int i = final_untied_count; i < final_untied_count + final_tied_count; i++) {
        z[i] = tied_final;
    }
}

std::tuple<int,int, bool> sum_largest_proj(double* z, int n, int k, double alpha, int untied, int tied, int cutoff,  bool debug) {
    const double TOL = 1e-9;
    const int MAX_ITERS = n;
    
    double val = compute_initial_sum(z, k);

    int final_untied_count;
    int final_tied_count;
    int iters = 0;
    
    double s1, s2, s;
    double untied_val, tied_val;
    std::pair<double, double> p;
    double untied_decrease = 0.0;
    double tied_final = z[k];
    double last_untied_val = z[k - 1];
    double post_tied_val = z[k];
    while ((val > alpha + TOL) && (iters < MAX_ITERS) && (tied + untied <= cutoff))
    {
        form_delta(untied, tied, k, n, p);
        double extra = val - alpha;
        if (debug) {
            print_debug_info(z, n, untied, tied, untied_decrease, tied_final, val, k);
        }

        untied_val = p.first;
        tied_val = p.second;

        compute_step_sizes(untied, tied, n, untied_val, tied_val, tied_final, 
                          last_untied_val, post_tied_val, s1, s2);

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
    }
    
    apply_final_projection(z, final_untied_count, final_tied_count, untied_decrease, tied_final);

    bool complete = (val <= alpha + TOL) ? true : false;
    std::tuple<int,int, bool> r = std::make_tuple(final_untied_count, final_tied_count, complete);
    return r;
}
