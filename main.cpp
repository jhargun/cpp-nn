#include <iostream>
#include <vector>

#include "matrix.h"
#include "mlp.h"
#include "activations/sigmoid.h"

using namespace std;

int main() {
    Sigmoid<float> sigmoid;
    MLP<float> mlp({2, 3, 1}, sigmoid);

    // XOR data (with some noise)
    float dataPoints[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
        0.1, 0.1, 0,
        0.1, 0.9, 1,
        0.9, 0.1, 1,
        0.9, 0.9, 0
    };
    vector<pair<Matrix<float>, Matrix<float>>> data;

    for (int i = 0; i < 8; ++i) {
        vector<float> inVec{dataPoints[i*3], dataPoints[i*3 + 1]};
        vector<float> outVec{dataPoints[i*3 + 2]};
        data.emplace_back(
            Matrix<float>{vector<vector<float>>{inVec}},
            Matrix<float>(vector<vector<float>>{outVec})
        );
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