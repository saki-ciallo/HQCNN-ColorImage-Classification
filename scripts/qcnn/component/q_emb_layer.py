import pennylane as qml
from pennylane import numpy as np

def emb_type1(inputs, num_qubit):
    # example: [[a00,a01],[a10,a11]] --> q0: Y_a00, Z_a10; q1: Y_a01, Z_a11
    # Channel 1
    qml.AngleEmbedding(inputs[:,0:2], wires=[0,1], rotation='Y')
    qml.AngleEmbedding(inputs[:,2:4], wires=[0,1], rotation='Z')
    # Channel 2
    qml.AngleEmbedding(inputs[:,4:6], wires=[2,3], rotation='Y')
    qml.AngleEmbedding(inputs[:,6:8], wires=[2,3], rotation='Z')
    # Channel 3
    qml.AngleEmbedding(inputs[:,8:10], wires=[4,5], rotation='Y')
    qml.AngleEmbedding(inputs[:,10:12], wires=[4,5], rotation='Z')
    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def emb_type2(inputs, num_qubit):
    # example: [[a00,a01],[a10,a11]] --> q0: Y_a00, Z_a10; q1: Y_a01, Z_a11
    # Channel 1
    qml.AngleEmbedding(inputs[:,0:2], wires=[0,1], rotation='X')
    qml.AngleEmbedding(inputs[:,2:4], wires=[0,1], rotation='Z')
    # Channel 2
    qml.AngleEmbedding(inputs[:,4:6], wires=[2,3], rotation='X')
    qml.AngleEmbedding(inputs[:,6:8], wires=[2,3], rotation='Z')
    # Channel 3
    qml.AngleEmbedding(inputs[:,8:10], wires=[4,5], rotation='X')
    qml.AngleEmbedding(inputs[:,10:12], wires=[4,5], rotation='Z')
    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

def emb_type3(inputs, num_qubit):
    # example: [[a00,a01],[a10,a11]] --> q0: Y_a00, Z_a10; q1: Y_a01, Z_a11
    # Channel 1
    qml.AngleEmbedding(inputs[:,0:2], wires=[0,1], rotation='X')
    qml.AngleEmbedding(inputs[:,2:4], wires=[0,1], rotation='Y')
    # Channel 2
    qml.AngleEmbedding(inputs[:,4:6], wires=[2,3], rotation='X')
    qml.AngleEmbedding(inputs[:,6:8], wires=[2,3], rotation='Y')
    # Channel 3
    qml.AngleEmbedding(inputs[:,8:10], wires=[4,5], rotation='X')
    qml.AngleEmbedding(inputs[:,10:12], wires=[4,5], rotation='Y')
    qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

# def emb_6qubit_type2(inputs, num_qubit):
#     # example: [[a00,a01],[a10,a11]] --> q0: Y_a00, Z_a01; q1: Y_a10, Z_a11
#     for idx in range(3):
#         qml.AngleEmbedding(inputs[:,0+idx*4:1+idx*4], wires=[idx*2], rotation='Y')
#         qml.AngleEmbedding(inputs[:,1+idx*4:2+idx*4], wires=[idx*2], rotation='Z')
#         qml.AngleEmbedding(inputs[:,2+idx*4:3+idx*4], wires=[1+idx*2], rotation='Y')
#         qml.AngleEmbedding(inputs[:,3+idx*4:4+idx*4], wires=[1+idx*2], rotation='Z')
#     qml.Barrier(wires=list(range(num_qubit)), only_visual=True)

# def emb3_4qubit(inputs):
#     # Channel 1
#     qml.AngleEmbedding(inputs[:,0:4], wires=range(4), rotation='X')
#     # Channel 2
#     qml.AngleEmbedding(inputs[:,4:8], wires=range(4), rotation='Y')
#     # Channel 3
#     qml.AngleEmbedding(inputs[:,8:12], wires=range(4), rotation='Z')
#     qml.Barrier(wires=list(range(4)), only_visual=True)