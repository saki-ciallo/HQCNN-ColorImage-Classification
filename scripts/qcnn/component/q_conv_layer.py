import pennylane as qml

def conv_layer1_type1(weights, num_qubit):
    # 6 params
    num_qubits = 6
    wires_pairs = [[i, i + 1] for i in range(0, num_qubits - 1, 2)]
    for i, wires in enumerate(wires_pairs):
        qml.IsingYY(weights[i], wires=wires)

    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[3, 4])

    for i, wires in enumerate(wires_pairs):
        qml.IsingZZ(weights[3 + i], wires=wires)

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def conv_layer2_type1(weights, num_qubit):
    # u3: 9 params, ising: 4 params
    # total: 13 params
    num_qubits = 4
    wires_pairs = [[i, i + 2] for i in range(0, num_qubits - 1, 2)]
    qml.U3(*weights[0:3], wires=[0])
    qml.U3(*weights[3:6], wires=[2])
    qml.U3(*weights[6:9], wires=[4])
    for i, wires in enumerate(wires_pairs):
        qml.IsingYY(weights[9 + i], wires=wires)

    for i, wires in enumerate(wires_pairs):
        qml.IsingZZ(weights[9 + 2 + i], wires=wires)

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def option_layer_type1(weights, num_qubit):
    qml.CRX(weights[0], wires=[0, 5])
    qml.CRX(weights[1], wires=[4, 1])
    qml.CRX(weights[2], wires=[2, 3])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def option_layer_type2(weights, num_qubit):
    qml.CRY(weights[0], wires=[0, 5])
    qml.CRY(weights[1], wires=[4, 1])
    qml.CRY(weights[2], wires=[2, 3])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def option_layer_type3(weights, num_qubit):
    qml.CRZ(weights[0], wires=[0, 5])
    qml.CRZ(weights[1], wires=[4, 1])
    qml.CRZ(weights[2], wires=[2, 3])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)





def conv_type1(weights, num_qubit):
    # Convolutional Layer 1
    # (num_qubits/2)*3 num of params
    num_qubits = 6
    wires_pairs = [[i, i + 1] for i in range(0, num_qubits - 1, 2)]
    print(wires_pairs)
    for i, wires in enumerate(wires_pairs):
        qml.IsingXX(weights[0 + 3 * i], wires=wires)
        qml.IsingYY(weights[1 + 3 * i], wires=wires)
        qml.IsingZZ(weights[2 + 3 * i], wires=wires)
    ising_params_num = 3 * len(wires_pairs)

    for i in range(num_qubits):
        qml.U3(*weights[ising_params_num+3*i : ising_params_num+3*(i+1)], wires=[i]) # 3*6
    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def conv_type2(weights, num_qubit):
    # Convolutional Layer 2
    num_qubits = 4
    wires_pairs = [[i, i + 2] for i in range(0, num_qubits - 1, 2)]
    for i, wires in enumerate(wires_pairs):
        qml.IsingXX(weights[0 + 9 * i], wires=wires)
        qml.IsingYY(weights[1 + 9 * i], wires=wires)
        qml.IsingZZ(weights[2 + 9 * i], wires=wires)
        qml.U3(*weights[3+9*i:6+9*i], wires=wires[0])
        qml.U3(*weights[6+9*i:9+9*i], wires=wires[1])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

