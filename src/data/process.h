#ifndef PROCESS_H
#define PROCESS_H

#include "loader.h"
#include <cmath>
#include <vector>

class DataProcessor {
public:
    // 处理缺失值
    static void handle_missing_values(Dataset* dataset) {
        if (!dataset) return;

        // 计算年龄平均值（忽略-1）
        double age_sum = 0.0;
        int age_count = 0;
        for (int i = 0; i < dataset->n_samples; i++) {
            if (dataset->data[i].features[AGE] >= 0) {
                age_sum += dataset->data[i].features[AGE];
                age_count++;
            }
        }
        double mean_age = age_count > 0 ? age_sum / age_count : 30.0;

        // 填充缺失值
        for (int i = 0; i < dataset->n_samples; i++) {
            // 年龄缺失用平均值填充
            if (dataset->data[i].features[AGE] < 0) {
                dataset->data[i].features[AGE] = mean_age;
            }
        }
    }

    // 标准化数据集
    static void normalize_dataset(Dataset* dataset) {
        if (!dataset) return;

        // 对每个特征分别处理
        for (int feature = 0; feature < dataset->n_features; feature++) {
            // 跳过不需要标准化的分类特征
            if (feature == SEX || feature == EMBARKED) {
                continue;  // 这些特征保持原值
            }

            // 对数值型特征进行标准化
            if (feature == PCLASS || feature == AGE || feature == SIBSP || 
                feature == PARCH || feature == FARE) {
                normalize_numeric_feature(dataset, feature);
            }
        }
    }

private:
    // 标准化数值型特征
    static void normalize_numeric_feature(Dataset* dataset, int feature_idx) {
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;

        // 计算均值和标准差
        for (int i = 0; i < dataset->n_samples; i++) {
            double value = dataset->data[i].features[feature_idx];
            if (value >= 0) {  // 忽略缺失值
                sum += value;
                sum_sq += value * value;
                count++;
            }
        }

        if (count > 0) {
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double std_dev = sqrt(variance);

            // 避免除以零
            if (std_dev > 0) {
                // 进行标准化 (z-score)
                for (int i = 0; i < dataset->n_samples; i++) {
                    if (dataset->data[i].features[feature_idx] >= 0) {
                        dataset->data[i].features[feature_idx] = 
                            (dataset->data[i].features[feature_idx] - mean) / std_dev;
                    }
                }
            }
        }
    }
};

#endif // PROCESS_H
