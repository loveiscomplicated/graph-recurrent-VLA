"""Input-only adapter for the existing three positional robot readout slots.

The pinned encoder reads robot rows0/1/2. Canonicalize those rows by their
already-present semantic node_type before passing to the unchanged encoder.
On every serialized benchmark graph this is the identity transform. It changes
neither parameters nor computations for valid original-order inputs.
"""
import torch
from torch import nn
from graph.integration_baseline.graph import PackedGraphs
from graph.persistent_geometry.encoder import DiagnosticEncoder
from graph.prototype.models import mlp

def canonicalize_robot_slots(graph):
    if bool(torch.all(graph.ptr[1:]-graph.ptr[:-1]>=3)):
        expected=graph.ptr[:-1,None]+torch.arange(3,device=graph.node_type.device)
        if torch.equal(graph.node_type[expected],torch.arange(3,device=graph.node_type.device).expand(graph.count,3)):
            return graph
    pieces=[]
    for owner in range(graph.count):
        rows=torch.nonzero(graph.node_graph==owner,as_tuple=True)[0]
        anchors=[]
        for kind in (0,1,2):
            found=rows[graph.node_type[rows]==kind]
            if len(found)!=1:raise ValueError('exactly one semantic robot anchor per role required')
            anchors.append(found)
        pieces.extend(anchors+[rows[graph.node_type[rows]>=3]])
    order=torch.cat(pieces)
    if torch.equal(order,torch.arange(len(order),device=order.device)):return graph
    inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
    return PackedGraphs(graph.node_input[order],graph.node_type[order],inverse[graph.edge_index],
        graph.edge_attr,graph.edge_type,graph.node_graph[order],graph.ptr,graph.frame_map)

class CanonicalDiagnosticEncoder(DiagnosticEncoder):
    def forward(self,graph):return super().forward(canonicalize_robot_slots(graph))

class CanonicalRegressor(nn.Module):
    def __init__(self,variant,output_dimension=36):
        super().__init__();self.encoder=CanonicalDiagnosticEncoder(variant)
        self.decoder=mlp([self.encoder.output_dimension,64,output_dimension])
    def forward(self,graph):return self.decoder(self.encoder(graph)[0])
