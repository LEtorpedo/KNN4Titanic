#ifndef ADAPTIVE_WEIGHTS_H
#define ADAPTIVE_WEIGHTS_H

#include <vector>
#include <cmath>
#include "../data/loader.h"

class AdaptiveWeights {
public:
    AdaptiveWeights(int n_features) 
        : weights_(n_features, 1.0),
          feature_success_(n_features, 0),
          feature_used_(n_features, 0) {}

    // 更新权重
    void update(const double* query, const std::vector<size_t>& neighbors,
                const Dataset* train_data, bool correct_prediction) {
        for (int f = 0; f < weights_.size(); f++) {
            bool feature_helpful = is_feature_helpful(f, query, neighbors, train_data);
            feature_used_[f]++;
            
            if (feature_helpful == correct_prediction) {
                feature_success_[f]++;
            }
            
            // 更新权重：成功率 * 2 + 0.5
            weights_[f] = (feature_success_[f] / (double)feature_used_[f]) * 2.0 + 0.5;
        }
    }

    const std::vector<double>& get_weights() const {
        return weights_;
    }

private:
    std::vector<double> weights_;
    std::vector<int> feature_success_;
    std::vector<int> feature_used_;

    bool is_feature_helpful(int feature_idx, const double* query,
                          const std::vector<size_t>& neighbors,
                          const Dataset* train_data) {
        // 计算查询点与邻居在该特征上的平均差异
        double avg_diff = 0.0;
        for (size_t idx : neighbors) {
            avg_diff += std::abs(query[feature_idx] - 
                      train_data->data[idx].features[feature_idx]);
        }
        avg_diff /= neighbors.size();
        
        // 如果差异小，认为该特征有帮助
        return avg_diff < 0.5;
    }
};

#endif // ADAPTIVE_WEIGHTS_H 