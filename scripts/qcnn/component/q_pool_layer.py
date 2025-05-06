import pennylane as qml

def pool_layer1_type1(weights, num_qubit):
    # Pooling Layer 1
    # 3 params
    qml.CPhase(weights[0], wires=[1,0])
    qml.CPhase(weights[1], wires=[3,2])
    qml.CPhase(weights[2], wires=[5,4])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def pool_layer2_type1(weights, num_qubit):
    # Pooling Layer 2
    # 2 params
    qml.CPhase(weights[0], wires=[4,0])
    qml.CPhase(weights[1], wires=[2,0])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)


def pool_layer2_type2(weights, num_qubit):
    # Pooling Layer 2
    # 1 params
    qml.CPhase(weights[0], wires=[2,0])

    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)




# def pool1_6qubit_type1(weights, num_qubit):
#     qml.CNOT(wires=[1,2])
#     qml.CNOT(wires=[3,4])
#     qml.CNOT(wires=[2,3])
#     # qml.CNOT(wires=[5,0])
#     # qml.Barrier(wires=list(range(num_wires)), only_visual=True)
#     qml.CPhase(weights[0], wires=[1,0])
#     qml.CPhase(weights[1], wires=[3,2])
#     qml.CPhase(weights[2], wires=[5,4])
#     qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

# def pool1_6qubit_type2(weights, num_qubit):
#     for i in range(3):
#         qml.CNOT(wires=[1+i,2+i])
#     qml.CNOT(wires=[i+2,1])
#     qml.CPhase(weights[0], wires=[1,0])
#     qml.CPhase(weights[1], wires=[3,2])
#     qml.CPhase(weights[2], wires=[5,4])
#     qml.Barrier(wires=list(range(num_qubit)), only_visual=True)