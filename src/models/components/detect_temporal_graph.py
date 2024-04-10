import torch
import clip
from PIL import Image
import os
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn import HypergraphConv
from torch_geometric.datasets import Planetoid
from src.models.components.diffusion import Diffusion
import json
from src.models.reinforcement_module import RlUserModule
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import pickle


class GAT(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class HyperGraph(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128, dropout=0.6):
        super(HyperGraph, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels,out_channels)
        self.dropout_p = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class RlGraph(nn.Module):
    def __init__(self,embedding_checkpoint_path
                 ):
        super().__init__()
        self.model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        # self.user_embedding = DiffusionModule()
        # self.embedding_checkpoint_path = '/data/zlt/python_code/fake-news-baselines/logs/train_diffusion_pol/runs/2024-01-13_11-54-10/checkpoints/last.ckpt'
        self.embedding_checkpoint_path = embedding_checkpoint_path
        # self.user_embedding.load_state_dict(torch.load(self.embedding_checkpoint_path))
        # user_embedding.eval()
        self.user_embedding = RlUserModule.load_from_checkpoint(self.embedding_checkpoint_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.temporal_gnn = GAT()
        self.structure_gnn = GAT()
        self.mh_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.hypergraph = HyperGraph()
        self.classifier = classifier(input_size=1280,num_class=2)
        self.sigmoid = nn.Sigmoid()
        
    @torch.no_grad()
    def cal_user_embedding(self,seq, seq_len):
        seq = torch.Tensor(seq).to(text_features.device)
        seq_len = torch.Tensor(seq_len).to(text_features.device)
        user_feature = self.user_embedding.net.model.predict(seq.long(), seq_len.long())
        if user_feature.shape[0] == 128:
            return user_feature
        return torch.mean(user_feature,dim=0)
    
    @torch.no_grad()
    def trans_propogation_graph(self,seq, seq_len, mask=1435):
        node_list = [0]
        node_list_id = [0]
        edge_index = []
        edge_index_id = []
        node_id = {
            0:0
        }
        hyper_edge_node = []
        hyper_edge_id = []
        temp_edge_num = 0
        for item in seq:
            temp_hyper_node = []
            for i in range(len(item)):
                if item[i]==mask:
                    break
                else:
                    temp_hyper_node.append(item[i])
            hyper_edge_node+=temp_hyper_node
            hyper_edge_id=hyper_edge_id+[temp_edge_num]*len(temp_hyper_node)
            temp_edge_num+=1
            
            for i in range(len(item)):
                if item[i]==0:
                    continue
                if item[i]==mask:
                    break
                if item[i] not in node_list:
                    node_list.append(item[i])
                    node_id[item[i]] = len(node_id)
                    # node_list_id.append(len(node_list_id))
                if (item[i-1],item[i]) not in edge_index:
                    edge_index.append([item[i-1],item[i]])
                    edge_index_id.append((node_id[item[i-1]],node_id[item[i]]))
        transpose_edge_index_id = list(map(list, zip(*edge_index_id)))
        nodes = torch.Tensor(node_list).to('cuda')
        transpose_edge_index_id = torch.Tensor(transpose_edge_index_id).to('cuda')
        user_feature = self.user_embedding.net.model.item_embeddings(nodes.long())
        # user_feature = self.user_embedding.net.model.predict(seq.long(), seq_len.long())
        
        hyper_edge_node = [node_id[item] for item in hyper_edge_node]
        hyper_edge_index = torch.Tensor([hyper_edge_node,hyper_edge_id]).to('cuda')
        return user_feature, transpose_edge_index_id.to(torch.int), hyper_edge_index.to(torch.int64)

    
    def trans_rl_propogation_graph(self, action_list):
        node_id = {}
        for item in action_list:
            for item_1 in item:
                if item_1[0] not in node_id.keys():
                    node_id[item_1[0]] = len(node_id.keys())
                if item_1[1] not in node_id.keys():
                    node_id[item_1[1]] = len(node_id.keys())
        id_node = {v: k for k, v in node_id.items()}
        
        sorted_id_node = {k: id_node[k] for k in sorted(id_node.keys())}
        node_list = [v  for k, v in node_id.items()]
        
        id_action_list = []
        # item_list = []
        for i in range(len(action_list)):

            temp_list  = action_list[:i+1]
            temp_item =[]
            for item in temp_list:
                temp_item=temp_item+item
            item_1 = [[node_id[t[0]],node_id[t[1]]] for t in temp_item]
            id_action_list.append(list(zip(*item_1)))
        
        return node_list,id_action_list
        
        
    def forward(self, text,image_path,user_path):
        #text,image_path = batch
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        text_features = self.model.encode_text(text)
        text_features = text_features.squeeze(0)
        if text_features.dim()==1:
            text_features = text_features.unsqueeze(0)
            
        image_features_list = []
        for i in range(len(image_path)):
            image = image_path[i]
            if image == '-1':
                image_features = text_features[i]
            else: 
                try:
                    Image.open(image)
                except:
                    image_features = text_features[i]
                    image_features_list.append(image_features)
                    continue
                image_features = self.preprocess(Image.open(image)).unsqueeze(0).to("cuda")
                image_features = self.model.encode_image(image_features)
                image_features = image_features.squeeze(0)
            image_features_list.append(image_features)
        # fusion all the image_features to one tensor
        image_features_batch = torch.stack(image_features_list)
        image_features_batch = image_features_batch.squeeze()
        if image_features_batch.dim()==1:
            image_features_batch = image_features_batch.unsqueeze(0)

        
        directed_graph_features_list = []
        hyper_graph_features_list = []
        propagation_feature_batch = []
        for i in range(len(user_path)):
            user = user_path[i]
            if user == '-1':
                seq  = [[0]+99*[1435]]
                seq_len = [1]
            else:
                with open(user, 'r') as file:
                    user = json.load(file)
                action_list = user['action_list']
            user_feature_list, edge_index_list = self.trans_rl_propogation_graph(action_list)
            user_embedding_list = self.user_embedding.net.user_embeddings(torch.tensor(user_feature_list).to(text_features.device))
            temropal_graph_features_list = []
            for edge_index in edge_index_list:
                edge_index = torch.Tensor(edge_index).to('cuda').to(torch.int)
                temropal_graph_features_list.append(gData(x=user_embedding_list, edge_index=edge_index))
            temporal_batch_graph = Batch.from_data_list(temropal_graph_features_list)
            
            temporal_x = self.temporal_gnn(temporal_batch_graph.x,temporal_batch_graph.edge_index)  
            temporal_out,mask = to_dense_batch(temporal_x, temporal_batch_graph.batch)
            temporal_graph_out= temporal_out.mean(dim=1)
            temporal_graph_out = self.mh_attn_1(temporal_graph_out,temporal_graph_out)
            
            structure_graph_features_list = []
            edge_index = torch.Tensor(edge_index_list[-1]).to('cuda').to(torch.int)
            structure_graph_features_list.append(gData(x=user_embedding_list, edge_index=edge_index)) 
            structure_batch_graph = Batch.from_data_list(structure_graph_features_list)
            structure_x = self.structure_gnn(structure_batch_graph.x,structure_batch_graph.edge_index)  
            structure_out,mask = to_dense_batch(structure_x, structure_batch_graph.batch)
            structure_all_out = structure_out.mean(dim=1).squeeze()
            
            # propagation_out = torch.cat((temporal_all_out,structure_all_out))
            propagation_out = self.mh_attn_1(temporal_graph_out,structure_all_out)
            propagation_feature_batch.append(propagation_out)
        
        propagation_feature_batch = torch.stack(propagation_feature_batch)  
            
        fused_features = torch.cat((text_features,image_features_batch,propagation_feature_batch),dim=1)   
        outputs = self.classifier(fused_features.to(torch.float32))
        logits = self.sigmoid(outputs)
        
        return logits
    
class classifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res