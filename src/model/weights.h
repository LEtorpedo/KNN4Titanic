#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "../data/loader.h"
#include <cmath>

class WeightCalculator {
public:
    // 默认权重
    static constexpr double DEFAULT_WEIGHTS[7] = {
        2.0,  // PCLASS
        3.0,  // SEX
        1.5,  // AGE
        1.0,  // SIBSP
        1.0,  // PARCH
        1.2,  // FARE
        0.5   // EMBARKED
    };

    // 使用默认权重
    static double* calculate_feature_weights(const Dataset* dataset) {
        double* weights = new double[dataset->n_features];
        weights[PCLASS] = 2.0;    // 舱位等级很重要
        weights[SEX] = 3.0;       // 性别是最重要的特征
        weights[AGE] = 1.5;       // 年龄有一定影响
        weights[SIBSP] = 1.0;     // 兄弟姐妹数影响较小
        weights[PARCH] = 1.0;     // 父母子女数影响较小
        weights[FARE] = 1.2;      // 票价有一定关联
        weights[EMBARKED] = 0.5;  // 登船港口影响最小
        return weights;
    }

    // 使用自定义权重
    static double* set_custom_weights(const double custom_weights[], int size) {
        if (size != FEATURE_COUNT) {
            printf("警告：权重数量不匹配，使用默认权重\n");
            return calculate_feature_weights(nullptr);
        }
        
        double* weights = new double[FEATURE_COUNT];
        for (int i = 0; i < FEATURE_COUNT; i++) {
            weights[i] = custom_weights[i];
        }
        return weights;
    }

    // 获取特征名称（用于显示）
    static const char* get_feature_name(int index) {
        static const char* feature_names[] = {
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
        };
        if (index >= 0 && index < FEATURE_COUNT) {
            return feature_names[index];
        }
        return "Unknown";
    }

    // 打印当前权重
    static void print_weights(const double weights[]) {
        printf("\n=== 当前特征权重 ===\n");
        for (int i = 0; i < FEATURE_COUNT; i++) {
            printf("%s: %.2f\n", get_feature_name(i), weights[i]);
        }
        printf("==================\n\n");
    }
};

#endif // WEIGHTS_H 