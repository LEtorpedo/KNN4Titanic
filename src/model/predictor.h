#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "kdtree.h"
#include "adaptive_weights.h"

class Predictor {
public:
    Predictor(const Dataset* train_data, const double* weights) 
        : train_data_(train_data),
          kdtree_(new KDTree(train_data)),
          static_weights_(weights),
          adaptive_weights_(nullptr),
          use_adaptive_(false) {}

    Predictor(const Dataset* train_data, bool use_adaptive = true) 
        : train_data_(train_data),
          kdtree_(new KDTree(train_data)),
          static_weights_(nullptr),
          adaptive_weights_(use_adaptive ? new AdaptiveWeights(train_data->n_features) : nullptr),
          use_adaptive_(use_adaptive) {}

    ~Predictor() {
        delete kdtree_;
        delete adaptive_weights_;
    }

    std::vector<int> predict(const Dataset* test_data, int k) {
        if (use_adaptive_) {
            return predict_adaptive(test_data, k);
        } else {
            return predict_static(test_data, k);
        }
    }

    std::vector<double> get_current_weights() const {
        if (use_adaptive_ && adaptive_weights_) {
            return adaptive_weights_->get_weights();
        } else {
            int n_features = train_data_->n_features;
            return std::vector<double>(static_weights_, static_weights_ + n_features);
        }
    }

    void predict_with_neighbors(const Dataset* test_data, int k,
                              std::vector<int>& predictions,
                              std::vector<std::vector<size_t>>& all_neighbors) {
        predictions.clear();
        all_neighbors.clear();
        predictions.reserve(test_data->n_samples);
        all_neighbors.reserve(test_data->n_samples);

        for (int i = 0; i < test_data->n_samples; i++) {
            auto neighbors = kdtree_->find_k_nearest(
                test_data->data[i].features, k, 
                use_adaptive_ ? adaptive_weights_->get_weights().data() : static_weights_);
            
            all_neighbors.push_back(neighbors);
            
            // 投票预测
            int survived_votes = 0;
            for (size_t idx : neighbors) {
                survived_votes += train_data_->data[idx].survived;
            }
            
            predictions.push_back(2 * survived_votes >= neighbors.size());
        }
    }

private:
    const Dataset* train_data_;
    KDTree* kdtree_;
    const double* static_weights_;
    AdaptiveWeights* adaptive_weights_;
    bool use_adaptive_;

    std::vector<int> predict_static(const Dataset* test_data, int k) {
        std::vector<int> predictions;
        predictions.reserve(test_data->n_samples);

        for (int i = 0; i < test_data->n_samples; i++) {
            auto neighbors = kdtree_->find_k_nearest(
                test_data->data[i].features, k, static_weights_);
            predictions.push_back(make_prediction(neighbors));
        }

        return predictions;
    }

    std::vector<int> predict_adaptive(const Dataset* test_data, int k) {
        std::vector<int> predictions;
        predictions.reserve(test_data->n_samples);

        for (int i = 0; i < test_data->n_samples; i++) {
            const auto& current_weights = adaptive_weights_->get_weights();
            
            auto neighbors = kdtree_->find_k_nearest(
                test_data->data[i].features, k, current_weights.data());
            
            int prediction = make_prediction(neighbors);
            predictions.push_back(prediction);

            if (test_data->data[i].survived != -1) {
                bool correct = (prediction == test_data->data[i].survived);
                adaptive_weights_->update(test_data->data[i].features, 
                                       neighbors, train_data_, correct);
            }
        }

        return predictions;
    }

    int make_prediction(const std::vector<size_t>& neighbors) {
        int survived_votes = 0;
        for (size_t idx : neighbors) {
            survived_votes += train_data_->data[idx].survived;
        }
        return 2 * survived_votes >= neighbors.size();
    }
};

#endif // PREDICTOR_H