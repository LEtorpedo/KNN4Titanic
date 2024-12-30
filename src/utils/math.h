#ifndef MATH_H
#define MATH_H

#include <math.h>

class MathUtils {
public:
    static double euclidean_distance(const double* p1, const double* p2, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    static double weighted_euclidean_distance(const double* p1, const double* p2, 
                                           int dim, const double* weights) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double diff = p1[i] - p2[i];
            sum += weights[i] * diff * diff;
        }
        return sqrt(sum);
    }
};

#endif // MATH_H 