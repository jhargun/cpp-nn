#include <iostream>
#include <vector>

#include "matrix.h"
#include "mlp.h"
#include "activations/sigmoid.h"
// #include "activations/activation.h"

using namespace std;

using Dtype = float;

int main() {
    Sigmoid<Dtype> fn = Sigmoid<Dtype>();
    MLP<Dtype> mlp({2, 3, 1}, fn);

    // // XOR data (with some noise)
    // Dtype dataPoints[] = {
    //     0, 0, 0,
    //     0, 1, 1,
    //     1, 0, 1,
    //     1, 1, 0,
    //     0.1, 0.1, 0,
    //     0.1, 0.9, 1,
    //     0.9, 0.1, 1,
    //     0.9, 0.9, 0
    // };
    // XOR data (with some noise)
    Dtype dataPoints[] = {
        1, 1, 1,
        0, 0, -1,
        .9, .9, 1,
        .1, .1, -1,
    };
    vector<pair<Matrix<Dtype>, Matrix<Dtype>>> data;

    for (int i = 0; i < (sizeof(dataPoints) / sizeof(dataPoints[0]) / 3); ++i) {
        vector<vector<Dtype>> inVec = {{dataPoints[i*3], dataPoints[i*3 + 1]}};
        vector<vector<Dtype>> outVec = {{dataPoints[i*3 + 2]}};
        Matrix<Dtype> inMat{inVec};
        Matrix<Dtype> outMat(outVec);
        data.emplace_back(inMat, outMat);
    }

    cout << "Initial preds:" << endl;
    for (const auto &pair : data) {
        cout << "pred: " << mlp.predict(pair.first)(0,0) << " vs correct: " << pair.second(0,0) << endl;
    }

    mlp.train(data, 0.01, 10000, false);

    cout << "Final preds:" << endl;
    for (const auto &pair : data) {
        cout << "pred: " << mlp.predict(pair.first)(0,0) << " vs correct: " << pair.second(0,0) << endl;
    }
}