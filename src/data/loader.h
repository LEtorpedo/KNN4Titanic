#ifndef LOADER_H
#define LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <sstream>
#include <vector>
#include <algorithm>

// 基础数据结构
typedef struct {
    double* features;
    int survived;
} Sample;

typedef struct {
    Sample* data;
    int n_samples;
    int n_features;
} Dataset;

// 特征索引枚举
enum TitanicFeatures {
    PCLASS,
    SEX,
    AGE,
    SIBSP,
    PARCH,
    FARE,
    EMBARKED,
    FEATURE_COUNT
};

class DataLoader {
private:
    // 添加列索引结构
    struct ColumnIndices {
        int pclass = -1;
        int sex = -1;
        int age = -1;
        int sibsp = -1;
        int parch = -1;
        int fare = -1;
        int embarked = -1;
        int survived = -1;
    };

    static ColumnIndices parse_header(const char* header) {
        ColumnIndices indices;
        std::vector<std::string> columns;
        std::string current;
        
        // 解析标题行
        for (const char* p = header; *p; p++) {
            if (*p == ',') {
                columns.push_back(current);
                current.clear();
            } else if (*p != '\r' && *p != '\n') {
                current += tolower(*p);
            }
        }
        if (!current.empty()) {
            columns.push_back(current);
        }

        // 查找每个列的位置
        for (size_t i = 0; i < columns.size(); i++) {
            const std::string& col = columns[i];
            if (col == "pclass") indices.pclass = i;
            else if (col == "sex") indices.sex = i;
            else if (col == "age") indices.age = i;
            else if (col == "sibsp") indices.sibsp = i;
            else if (col == "parch") indices.parch = i;
            else if (col == "fare") indices.fare = i;
            else if (col == "embarked") indices.embarked = i;
            else if (col == "survived") indices.survived = i;
        }

        return indices;
    }

public:
    static Dataset* load_csv(const char* filename, bool is_training) {
        FILE* file = fopen(filename, "r");
        if (!file) {
            printf("无法打开文件: %s\n", filename);
            return NULL;
        }

        Dataset* dataset = new Dataset();
        dataset->n_features = FEATURE_COUNT;
        
        // 读取并解析标题行
        char header[1024];
        if (!fgets(header, sizeof(header), file)) {
            fclose(file);
            delete dataset;
            return NULL;
        }
        
        ColumnIndices col_idx = parse_header(header);
        
        // 计算样本数量
        dataset->n_samples = count_samples(file) - 1;
        dataset->data = new Sample[dataset->n_samples];

        // 重置文件指针到第二行
        rewind(file);
        fgets(header, sizeof(header), file); // 跳过标题行

        // 读取数据
        int row = 0;
        char line[1024];
        while (fgets(line, sizeof(line), file) && row < dataset->n_samples) {
            parse_line(line, col_idx, &dataset->data[row]);
            row++;
        }

        fclose(file);
        return dataset;
    }

    static void free_dataset(Dataset* dataset) {
        if (!dataset) return;
        
        for (int i = 0; i < dataset->n_samples; i++) {
            delete[] dataset->data[i].features;
        }
        delete[] dataset->data;
        delete dataset;
    }

private:
    static int count_samples(FILE* file) {
        int count = 0;
        char line[1024];
        while (fgets(line, sizeof(line), file)) {
            count++;
        }
        return count;
    }

    static double safe_stod(const std::string& str, double default_value = 0.0) {
        if (str.empty() || str == "\"\"") return default_value;
        try {
            return std::stod(str);
        } catch (...) {
            return default_value;
        }
    }

    static void parse_line(const char* line, const ColumnIndices& col_idx, Sample* sample) {
        sample->features = new double[FEATURE_COUNT];
        for (int i = 0; i < FEATURE_COUNT; i++) {
            sample->features[i] = 0.0;
        }

        std::vector<std::string> tokens;
        std::string current;
        bool in_quotes = false;
        
        // 更安全的CSV解析
        for (const char* p = line; *p; p++) {
            if (*p == '"') {
                in_quotes = !in_quotes;
            } else if (*p == ',' && !in_quotes) {
                tokens.push_back(current);
                current.clear();
            } else if (*p != '\r' && *p != '\n') {
                current += *p;
            }
        }
        if (!current.empty()) {
            tokens.push_back(current);
        }

        // 调试输出：显示列索引
        static bool first_time = true;
        if (first_time) {
            printf("\n=== 列索引信息 ===\n");
            printf("Pclass: %d\n", col_idx.pclass);
            printf("Sex: %d\n", col_idx.sex);
            printf("Age: %d\n", col_idx.age);
            printf("SibSp: %d\n", col_idx.sibsp);
            printf("Parch: %d\n", col_idx.parch);
            printf("Fare: %d\n", col_idx.fare);
            printf("Embarked: %d\n", col_idx.embarked);
            printf("Survived: %d\n", col_idx.survived);
            printf("================\n\n");
            first_time = false;
        }

        try {
            // 使用列索引读取数据
            if (col_idx.pclass >= 0 && col_idx.pclass < tokens.size())
                sample->features[PCLASS] = safe_stod(tokens[col_idx.pclass]);
            
            if (col_idx.sex >= 0 && col_idx.sex < tokens.size()) {
                std::string sex = tokens[col_idx.sex];
                // 移除引号和空格
                sex.erase(std::remove(sex.begin(), sex.end(), '"'), sex.end());
                sex.erase(std::remove(sex.begin(), sex.end(), ' '), sex.end());
                std::transform(sex.begin(), sex.end(), sex.begin(), ::tolower);
                // 使用精确匹配而不是部分匹配
                sample->features[SEX] = (sex == "male") ? 1.0 : 0.0;
            }
            
            if (col_idx.age >= 0 && col_idx.age < tokens.size())
                sample->features[AGE] = safe_stod(tokens[col_idx.age], -1.0);
            
            if (col_idx.sibsp >= 0 && col_idx.sibsp < tokens.size())
                sample->features[SIBSP] = safe_stod(tokens[col_idx.sibsp]);
            
            if (col_idx.parch >= 0 && col_idx.parch < tokens.size())
                sample->features[PARCH] = safe_stod(tokens[col_idx.parch]);
            
            if (col_idx.fare >= 0 && col_idx.fare < tokens.size())
                sample->features[FARE] = safe_stod(tokens[col_idx.fare]);
            
            if (col_idx.embarked >= 0 && col_idx.embarked < tokens.size())
                sample->features[EMBARKED] = parse_embarked(tokens[col_idx.embarked]);
            
            if (col_idx.survived >= 0 && col_idx.survived < tokens.size())
                sample->survived = static_cast<int>(safe_stod(tokens[col_idx.survived], -1.0));
            else
                sample->survived = -1;

            

        } catch (const std::exception& e) {
            printf("解析错误: %s\n行内容: %s\n", e.what(), line);
            // 保持默认值
        }
    }

    static double parse_embarked(const std::string& embarked) {
        if (embarked.empty() || embarked == "NA") return 0.0;
        char port = embarked[0];
        switch (port) {
            case 'S': return 0.0;
            case 'C': return 1.0;
            case 'Q': return 2.0;
            default: return 0.0;
        }
    }
};

#endif // LOADER_H
