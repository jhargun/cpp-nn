#include <iostream>
#include <vector>

#include "matrix.h"
#include "mlp.h"
#include "activations/sigmoid.h"
// #include "activations/activation.h"

using namespace std;

using Dtype = float;

int main() {
    MLP<Dtype> mlp({2, 3, 1}, Sigmoid<Dtype>());

    // XOR data (with some noise)
    Dtype dataPoints[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
        0.1, 0.1, 0,
        0.1, 0.9, 1,
        0.9, 0.1, 1,
        0.9, 0.9, 0
    };
    vector<pair<Matrix<Dtype>, Matrix<Dtype>>> data;

    for (int i = 0; i < 8; ++i) {
        vector<vector<Dtype>> inVec = {{dataPoints[i*3], dataPoints[i*3 + 1]}};
        vector<vector<Dtype>> outVec = {{dataPoints[i*3 + 2]}};
        Matrix<Dtype> inMat{inVec};
        Matrix<Dtype> outMat(outVec);
        data.emplace_back(inMat, outMat);
    }

    cout << "Initial preds:" << endl;
    for (const auto &pair : data) {
        cout << "pred: " << mlp.predict(pair.first) << " vs correct: " << pair.second << endl;
    }

    mlp.train(data, 0.05, 20, true);

    cout << "Final preds:" << endl;
    for (const auto &pair : data) {
        cout << "pred: " << mlp.predict(pair.first) << " vs correct: " << pair.second << endl;
    }
}