import torch
import clip
from PIL import Image
import os
from torch import nn
from torch.distributions import Categorical
import json
import torch.nn.functional as F

class RlNet(nn.Module):
    def __init__(self,user_num=114516,hidden_size=128
                 ):
        super().__init__()
        self.model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = classifier(input_size=512,num_class=2)
        self.sigmoid = nn.Sigmoid()
        
        self.init_nn =  nn.Linear(512,128)
        
        self.user_embeddings = nn.Embedding(
            num_embeddings=user_num + 1,
            embedding_dim=hidden_size,
        )
        self.action_adj_path = './RL_data/politifact/all_actions_politifact.json'
        with open(self.action_adj_path , 'r') as f:
            self.action_adj = json.load(f)
        
        self.MAX_DEPTH = 4
        self.MAX_ACTION = 50
        self.config={}
        self.config['embedding_size']=128
        self.softmax = nn.Softmax(dim=-2)
        self.node_view_gnn, self.meta_view_gnn_1, self.meta_view_gnn_2, self.meta_view_gnn_3 = GAT(), GAT(), GAT(), GAT()
        self.anchor_embedding_layer = nn.Sequential(
                                        nn.Linear(self.config['embedding_size']*2, self.config['embedding_size'], bias=False),
                                        nn.Tanh(),
                                    )
        self.anchor_layer = nn.Sequential(
                                nn.Linear(self.config['embedding_size'], self.config['embedding_size'], bias=False),
                                nn.ELU(),
                                nn.Linear(self.config['embedding_size'], 1, bias=False),
                            )
        self.policy_net = Net()
        
    def forward(self, id,text,actions,split_actions):
        
        padded_node,  negative_node = self.get_node_view_node_sample(actions)
        padded_node,  negative_node = self.get_meta_view_node_sample(actions, meta_path=['PNP','UNU','PNU'])
        con_loss_1 = self.cal_cl_loss(padded_node,  negative_node)
        con_loss_2 = self.cal_cl_loss(padded_node,  negative_node)

        #text,image_path = batch
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        text_features = self.model.encode_text(text)
        text_features = text_features.squeeze(0)
        if text_features.dim()==1:
            text_features = text_features.unsqueeze(0)
        text_embedding =  self.init_nn(text_features.to(torch.float32))
        
        # state embedding 
        temp_1 = torch.unsqueeze(torch.tensor([0]),dim=0).to(text_embedding.device)
        init_embeding = self.user_embeddings(temp_1.repeat(len(id),1))
        
        state_input = torch.cat([text_embedding,init_embeding.squeeze(1)], dim=-1)
        
        temp_2 = torch.unsqueeze(torch.tensor([i for i in range(1, 11)]),dim=0).to(text_embedding.device)
        action_id = temp_2.repeat(len(id),1)
        tail_user_embeding = self.user_embeddings(action_id)
        
        temp_3 = [(0,i) for i in range(1, 11)]
        action_list = [temp_3[:] for _ in range(len(id))]
        # init+tail
        action_embedding = tail_user_embeding
        
        depth = 0
        act_probs_steps = []
        state_values_steps = []
        propagation_graph = [temp_1.repeat(len(id),1)]
        rewards_steps = []
        all_select_action_list=[]
        while (depth < self.MAX_DEPTH):
        # 计算action概率
            act_probs, state_values = self.policy_net(state_input, action_embedding)
            top_k = 5
            select_act_probs, select_nodes, _, acts_idx = self.get_anchor_nodes(act_probs,action_id,action_embedding,top_k)
            # acts_idx = acts_idx.unsequene(1).to_item()
            select_action_list = []
            for i in range(len(action_list)):
                temp = []
                for j in acts_idx[i]:
                    length = j.item()
                    if length>=len(action_list[i]):
                        continue
                    else:
                        temp.append(action_list[i][length])
                                   
                select_action_list.append(temp)
                        
            depth = depth + 1
            act_probs_steps.append(select_act_probs)
            state_values_steps.append(state_values)
            propagation_graph.append(select_nodes)
            all_select_action_list.append(select_action_list)
            
            # 更新state 获取 action
            state_input = self.get_state_input(text_embedding,propagation_graph)
            action_embedding,action_id,_,action_list = self.get_action_embedding(select_nodes)
            
            # 返回reward
            reward = self.get_reward(actions,select_nodes)
            rewards_steps.append(reward)
        
        return act_probs_steps, state_values_steps, rewards_steps, propagation_graph, all_select_action_list, con_loss_1, con_loss_2
        
    def get_state_input(self, news_embedding, anchor_graph):
        # anchor_embedding = self.get_anchor_graph_embedding(anchor_graph)
        
        anchor_graph_nodes  = torch.cat(anchor_graph, dim=-1).to(news_embedding.device)
        anchor_embedding = self.user_embeddings(anchor_graph_nodes).to(news_embedding.device)

        # anchor_embedding = self.anchor_embedding_layer(anchor_embedding)#(batch, 50, 128)
        anchor_embedding_weight = self.softmax(self.anchor_layer(anchor_embedding))#(batch, 50, 1)
        anchor_embedding = torch.sum(anchor_embedding * anchor_embedding_weight, dim=-2)
        
        state_embedding = torch.cat([news_embedding, anchor_embedding], dim=-1)
        return state_embedding
    
    def get_node_view_node_sample(self,actions,max_len=20,max_sample_len=200):
        def pad_sequences(data, max_len, pad_value=0):
            padded_data = []
            for seq in data:
                if len(seq) >= max_len:
                    padded_seq = seq[:max_len]
                else:
                    padded_seq = seq + [pad_value] * (max_len - len(seq))
                padded_data.append(padded_seq)
            return padded_data
        actions_new = []
        all_node = []
        for item in actions:
            temp = []
            item = eval(item)
            for item_1 in item:
                
                temp.append(int(item_1['head_user']))
                temp.append(int(item_1['user']))
            actions_new.append(list(set(temp)))
            all_node = all_node +list(set(temp))
        padded_data = pad_sequences(actions_new, max_len)
        all_node = all_node[:max_sample_len]
        return padded_data,  all_node
    
    def cal_cl_loss(self, padded_node,  negative_node):
        padded_node = torch.tensor(padded_node).to('cuda')
        negative_node = torch.tensor(negative_node).to('cuda')
        
        padded_node_embeding = self.user_embeddings(padded_node)
        
        
        user_feature_list, edge_index_list = self.trans_propogation_graph(actions)
        user_embedding_list = self.user_embeddings(torch.tensor(user_feature_list).to('cuda'))
        user_embedding_list = self.pgd_attack(padded_node_embeding)
        
        ## node_view
        graph_features_list = []
        for edge_index in edge_index_list:
            edge_index = torch.Tensor(edge_index).to('cuda').to(torch.int)
            graph_features_list.append(gData(x=user_embedding_list, edge_index=edge_index))
        batch_graph = Batch.from_data_list(graph_features_list)

        x = self.gnn(batch_graph.x,batch_graph.edge_index)  
        embedding_out,mask = to_dense_batch(x, batch_graph.batch)
        
        all_sample_embeding = self.user_embeddings(negative_node)
        temp_similarity = embedding_out@all_sample_embeding.T
        
        topk=10
        max_values, max_indices = torch.topk(temp_similarity, k=topk, dim=-1, largest=True)
        min_values, min_indices = torch.topk(temp_similarity, k=topk, dim=-1, largest=False)
        
        pos_similarity_list = []
        neg_similarity_list = []
        
        for i in range(padded_node.shape[0]):
            temp_node_embedding = padded_node_embeding[i]
            temp_max_index = max_indices[i]
            temp_min_index = min_indices[i]
            postive_nodes = all_node[temp_max_index]
            negative_nodes = all_node[temp_max_index]
            positive_node_embedding = self.user_embeddings(postive_nodes)
            negative_node_embedding = self.user_embeddings(negative_nodes)
            pos_similarity_matrix = torch.matmul(temp_node_embedding.unsqueeze(1), positive_node_embedding.transpose(1, 2)).squeeze()
            neg_similarity_matrix = torch.matmul(temp_node_embedding.unsqueeze(1), negative_node_embedding.transpose(1, 2)).squeeze()
            pos_similarity_list.append(pos_similarity_matrix)
            neg_similarity_list.append(neg_similarity_matrix)
        batch_pos_similarity = torch.stack(pos_similarity_list, dim=0)
        batch_neg_similarity = torch.stack(neg_similarity_list, dim=0)
        
        con_loss = self.simclr_loss(batch_pos_similarity, batch_neg_similarity)
        
        return loss   
        

    def get_reward(self,actions,select_nodes):
        actions_new = []
        for item in actions:
            temp = []
            item = eval(item)
            for item_1 in item:
                
                temp.append(item_1['head_user'])
                temp.append(item_1['user'])
            actions_new.append(list(set(temp)))
        hit_rewards = torch.zeros([len(actions), len(select_nodes[0])], dtype=torch.float32).to(select_nodes.device)
        for i in range(len(actions)):
            for j in range(len(select_nodes[i])):
                idx = select_nodes[i][j].item()
                if idx in actions_new[i]:
                    hit_rewards[i][j] = 1.0
        return hit_rewards
                
        
    
    def get_action_embedding(self,select_nodes):
        # print(1)
        node_list = []
        action_list = []
        for i in range(len(select_nodes.tolist())):
            item = select_nodes.tolist()[i]
            temp_node_list = []
            temp_action_list = []
            for node in item:
                if str(node) not in self.action_adj.keys():
                    continue
                temp = [int(num) for num in self.action_adj[str(node)]]
                temp_node_list = temp_node_list+temp
                temp_action_list= temp_action_list + [(node, num) for num in temp]
            if(len(temp_node_list))==0:
                temp_node_list = [0]
                temp_action_list = [(0,0)]
            node_list.append(temp_node_list)
            action_list.append(temp_action_list)
        padded_array = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in node_list], batch_first=True, padding_value=0).to(select_nodes.device)
        truncated_array = padded_array[:, :self.MAX_ACTION].to(select_nodes.device)
        mask = (truncated_array != 0).float().to(select_nodes.device)
        return self.user_embeddings(truncated_array), truncated_array,mask, action_list
                
            
        
    
    def get_anchor_nodes(self, weights, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1)
            action_id_input = torch.unsqueeze(action_id_input, 1)
            relation_id_input = torch.unsqueeze(relation_id_input, 1)

        weights = weights.squeeze(-1)
        m = Categorical(weights)
        acts_idx = m.sample(sample_shape=torch.Size([topk]))#may sample the same position multiple times
        acts_idx = acts_idx.permute(1,2,0)
        shape0 = acts_idx.shape[0]
        shape1 = acts_idx.shape[1]
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2])#(batch,topk)

        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])
        # relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])

        # weights = weights.to('cpu')
        # acts_idx = acts_idx.to('cpu')
        weights = weights.gather(1, acts_idx)
        # action_id_input = action_id_input.to('cpu')
        state_id_input_value = action_id_input.gather(1, acts_idx)#selected entity id,(batch,topk)
        # relation_id_selected = relation_id_input.gather(1, acts_idx)#selected relation id,(batch,topk)
        
        # weights = weights.to(relation_id_input.device)
        # state_id_input_value = state_id_input_value.to(relation_id_input.device)
        
        weights = weights.reshape(shape0, shape1 *  weights.shape[1])#probility for selected (r,e) ,(batch,5), (batch, 15) , (batch, 30)
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 *  state_id_input_value.shape[1])
        # relation_id_selected = relation_id_selected.reshape(shape0, shape1 *  relation_id_selected.shape[1])
        
        relation_id_selected = None
        return weights, state_id_input_value, relation_id_selected, acts_idx
    
    def trans_propogation_graph(self, action_list):
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
    
    def simclr_loss(self,positive_similarity, negative_similarity, temperature=0.5):
        positive_similarity = torch.flatten(positive_similarity, start_dim=1)
        positive_similarity /= temperature  
        positive_log_prob = F.log_softmax(positive_similarity, dim=1)
        positive_loss = -torch.mean(torch.sum(positive_log_prob * positive_similarity, dim=1))


        negative_similarity = torch.flatten(negative_similarity, start_dim=1)
        negative_similarity /= temperature  
        negative_log_prob = F.log_softmax(negative_similarity, dim=1)
        negative_loss = -torch.mean(torch.sum(negative_log_prob * negative_similarity, dim=1))


        loss = positive_loss + negative_loss
        return loss
    
    def pgd_attack(self, node_vector, epsilon=0.1, alpha=0.01, num_steps=10):
        gradient = node_vector.grad
        perturbed_node_vector = node_vector.clone().detach()

        for _ in range(num_steps):

            perturbation = alpha * torch.sign(gradient)

            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            perturbed_node_vector += perturbation

            perturbed_node_vector = torch.clamp(perturbed_node_vector, -1, 1) 


            perturbed_node_vector.requires_grad = True
            loss = torch.sum(perturbed_node_vector * gradient)
            loss.backward()
            gradient = perturbed_node_vector.grad.data

            perturbed_node_vector.grad.zero_()

        return perturbed_node_vector


class Net(nn.Module):
    def __init__(self, doc_feature_embedding=128):
        super(Net, self).__init__()
        self.config={}
        self.config['embedding_size']=128
        self.doc_feature_embedding = doc_feature_embedding


        self.actor_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.actor_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.actor_l3 = nn.Linear(self.config['embedding_size'],1)

        #self.critic_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.critic_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.critic_l3 = nn.Linear(self.config['embedding_size'], 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2], state_input.shape[3])

        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.softmax(self.actor_l3(actor_out))#out: (batch, 20, 1),(batch, 5, 20, 1),(batch, 15, 20, 1)

        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        values = self.critic_l3(critic_out).mean(dim=-2)#out: (batch,1), (batch,5,1), (batch,15,1)

        return act_probs, values
 
class classifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(512,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)