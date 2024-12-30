#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "../utils/math.h"
#include "../data/loader.h"

// 前向声明
class KDTree;

struct NearestNeighbor {
    double distance;
    const double* point;
    size_t index;
    
    NearestNeighbor() : distance(INFINITY), point(nullptr), index(0) {}
};

struct KDNode {
    const double* point;
    size_t index;
    int split_dim;
    KDNode *left, *right;
    
    KDNode(const double* p, size_t idx, int dim) 
        : point(p), index(idx), split_dim(dim), left(nullptr), right(nullptr) {}
};

class KDTree {
private:
    const Dataset* dataset_;
    KDNode* root;

    // 声明所有私有成员函数
    void free_tree(KDNode* node);
    KDNode* build_tree(size_t* indices, size_t n_points, int depth);
    void find_k_nearest_impl(KDNode* node, const double* query, int k, 
                           NearestNeighbor* neighbors, const double* weights);

public:
    // 构造函数
    KDTree(const Dataset* dataset) : dataset_(dataset), root(nullptr) {
        if (!dataset || dataset->n_samples == 0) return;
        
        std::vector<size_t> indices(dataset->n_samples);
        for (size_t i = 0; i < dataset->n_samples; i++) {
            indices[i] = i;
        }
        
        root = build_tree(indices.data(), dataset->n_samples, 0);
    }

    // 析构函数
    ~KDTree() {
        free_tree(root);
    }

    // 公共接口
    std::vector<size_t> find_k_nearest(const double* query, int k, const double* weights) {
        std::vector<NearestNeighbor> neighbors(k);
        find_k_nearest_impl(root, query, k, neighbors.data(), weights);
        
        std::vector<size_t> result;
        for (const auto& neighbor : neighbors) {
            if (neighbor.point) {
                result.push_back(neighbor.index);
            }
        }
        return result;
    }
};

// 在类外定义私有成员函数
void KDTree::free_tree(KDNode* node) {
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    delete node;
}

KDNode* KDTree::build_tree(size_t* indices, size_t n_points, int depth) {
    if (n_points == 0) return nullptr;

    int split_dim = depth % dataset_->n_features;
    
    size_t mid = n_points / 2;
    std::nth_element(indices, indices + mid, indices + n_points,
        [this, split_dim](size_t a, size_t b) {
            return dataset_->data[a].features[split_dim] < 
                   dataset_->data[b].features[split_dim];
        });

    const double* point = dataset_->data[indices[mid]].features;
    KDNode* node = new KDNode(point, indices[mid], split_dim);

    node->left = build_tree(indices, mid, depth + 1);
    node->right = build_tree(indices + mid + 1, n_points - mid - 1, depth + 1);

    return node;
}

void KDTree::find_k_nearest_impl(KDNode* node, const double* query, int k, 
                               NearestNeighbor* neighbors, const double* weights) {
    if (!node) return;

    double dist = weights ? 
        MathUtils::weighted_euclidean_distance(query, node->point, dataset_->n_features, weights) :
        MathUtils::euclidean_distance(query, node->point, dataset_->n_features);

    int insert_pos = k - 1;
    while (insert_pos >= 0 && (neighbors[insert_pos].point == nullptr || 
           dist < neighbors[insert_pos].distance)) {
        if (insert_pos < k - 1) {
            neighbors[insert_pos + 1] = neighbors[insert_pos];
        }
        insert_pos--;
    }
    insert_pos++;
    
    if (insert_pos < k) {
        neighbors[insert_pos].distance = dist;
        neighbors[insert_pos].point = node->point;
        neighbors[insert_pos].index = node->index;
    }

    double split_dist = query[node->split_dim] - node->point[node->split_dim];
    if (weights) {
        split_dist *= weights[node->split_dim];
    }

    KDNode* first = split_dist <= 0 ? node->left : node->right;
    KDNode* second = split_dist <= 0 ? node->right : node->left;

    find_k_nearest_impl(first, query, k, neighbors, weights);

    if (neighbors[k-1].point == nullptr || 
        std::abs(split_dist) < neighbors[k-1].distance) {
        find_k_nearest_impl(second, query, k, neighbors, weights);
    }
}

#endif // KDTREE_H