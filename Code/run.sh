#!/bin/bash
mode="$1"
size="$2"
perceptive="$3"
cnn="$4"

echo ""
echo "================================================================"
echo "========================== Q-Learning =========================="
echo "================================================================"
echo ""
./virt-env/bin/python3 ./q-learning/q-learning.py "$mode" "$size"

echo ""
echo "================================================================"
echo "============================= SARSA ============================"
echo "================================================================"
echo ""
./virt-env/bin/python3 ./sarsa/sarsa.py "$mode" "$size"

echo ""
echo "================================================================"
echo "======================== Deep Q-Learning ======================="
echo "================================================================"
echo ""
./virt-env/bin/python3 ./deep-q-learning/deep-q-learning.py "$mode" "$size"

echo ""
echo "================================================================"
echo "================= Proximal Policy Optimization ================="
echo "================================================================"
echo ""
./virt-env/bin/python3 ./proximal-policy-optimization/proximal-policy-optimization.py "$mode" "$size"

if [[ "$perceptive" = "yes" ]]; then
    echo ""
    echo "================================================================"
    echo "================== Perceptive Deep Q-Learning =================="
    echo "================================================================"
    echo ""
    ./virt-env/bin/python3 ./perceptive-deep-q-learning/perceptive-deep-q-learning.py "$mode" "$size"

    echo ""
    echo "================================================================"
    echo "============ Perceptive Proximal Policy Optimization ==========="
    echo "================================================================"
    echo ""
    ./virt-env/bin/python3 ./perceptive-proximal-policy-optimization/perceptive-proximal-policy-optimization.py "$mode" "$size"
fi

if [[ "$cnn" = "yes" ]]; then
    echo ""
    echo "================================================================"
    echo "===== Perceptive Proximal Policy Optimization (CNN Variant) ===="
    echo "================================================================"
    echo ""
    ./virt-env/bin/python3 ./perceptive-proximal-policy-optimization/perceptive-proximal-policy-optimization-cnn.py "$mode" "$size"
fi
