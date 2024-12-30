#include "data/loader.h"
#include "data/process.h"
#include "model/predictor.h"
#include "model/weights.h"
#include <chrono>
#include <fstream>
#include <iomanip>

// 用于性能计时的宏
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define DURATION(start) \
    std::chrono::duration_cast<std::chrono::milliseconds>(TIME_NOW - start).count()

// 计算准确率
double calculate_accuracy(const std::vector<int>& predictions, const char* truth_file) {
    std::ifstream file(truth_file);
    if (!file.is_open()) {
        printf("无法打开真实值文件: %s\n", truth_file);
        return 0.0;
    }

    // 跳过标题行
    std::string line;
    std::getline(file, line);

    int correct = 0, total = 0;
    int id, truth;
    char comma;
    while (file >> id >> comma >> truth) {
        if (total < predictions.size() && predictions[total] == truth) {
            correct++;
        }
        total++;
    }

    return total > 0 ? (double)correct / total : 0.0;
}

int main() {
    printf("=== 泰坦尼克号生存预测 ===\n\n");

    auto total_start = TIME_NOW;

    // 1. 加载数据
    auto load_start = TIME_NOW;
    Dataset* train_data = DataLoader::load_csv("../data/train.csv", true);
    Dataset* test_data = DataLoader::load_csv("../data/test.csv", false);
    printf("数据加载耗时: %ldms\n", DURATION(load_start));

    if (!train_data || !test_data) {
        printf("数据加载失败\n");
        return 1;
    }

    // 2. 数据预处理
    auto preprocess_start = TIME_NOW;
    DataProcessor::handle_missing_values(train_data);
    DataProcessor::handle_missing_values(test_data);
    DataProcessor::normalize_dataset(train_data);
    DataProcessor::normalize_dataset(test_data);
    printf("数据预处理耗时: %ldms\n", DURATION(preprocess_start));

    // 3. 特征权重计算
    auto weight_start = TIME_NOW;
    double custom_weights[] = {
        2.0,  // Pclass
        3.0,  // Sex
        1.5,  // Age
        1.0,  // SibSp
        1.0,  // Parch
        1.2,  // Fare
        0.5   // Embarked
    };
    
    double* weights = WeightCalculator::set_custom_weights(custom_weights, FEATURE_COUNT);
    printf("特征权重计算耗时: %ldms\n", DURATION(weight_start));

    // 4. 模型训练
    auto train_start = TIME_NOW;
    Predictor predictor(train_data, weights);
    printf("模型训练耗时: %ldms\n", DURATION(train_start));

    // 5. 预测
    auto predict_start = TIME_NOW;
    std::vector<int> predictions;
    std::vector<std::vector<size_t>> all_neighbors;
    predictor.predict_with_neighbors(test_data, 5, predictions, all_neighbors);
    printf("预测耗时: %ldms\n", DURATION(predict_start));

    // 6. 计算准确率
    double accuracy = calculate_accuracy(predictions, "../data/gender_submission.csv");
    printf("\n预测准确率: %.2f%%\n", accuracy * 100);
    printf("总耗时: %ldms\n", DURATION(total_start));

    // 保存预测结果
    std::ofstream out_file("predictions.csv");
    if (out_file.is_open()) {
        out_file << "PassengerId,Survived\n";
        for (size_t i = 0; i < predictions.size(); i++) {
            out_file << (i + 892) << "," << predictions[i] << "\n";
        }
    }

    // 清理内存
    delete[] weights;
    DataLoader::free_dataset(train_data);
    DataLoader::free_dataset(test_data);

    return 0;
} 