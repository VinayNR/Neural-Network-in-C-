#include<iostream>
#include<cmath>
#include<vector>

using namespace std;

int no_of_prev_nodes;

typedef struct Node1
{
    float bias;
    vector<float> weight;
    struct Node1 * next;
}node;

typedef struct Node2
{
    int no_of_nodes;
    node * nodes;
    struct Node2 * next;
}layer;

typedef struct Node3
{
    int no_of_layers;
    layer * hidden_layers;
    layer * output_layer;
}graph;

layer * create_hidden_layer(int no_of_nodes)
{
    layer * temp = new layer;
    temp->no_of_nodes = no_of_nodes;
    temp->nodes = NULL;
    temp->next = NULL;
    return temp;
}

node * create_node(float b, vector<float> & w)
{
    node * temp = new node;
    temp->bias = b;
    for(int i=0;i<w.size();i++)
        temp->weight.push_back(w[i]);
    temp->next = NULL;
    return temp;
}

void add_node_to_layer(node * node_temp, layer * hidden_temp)
{
    if(hidden_temp->nodes == NULL)
        hidden_temp->nodes = node_temp;
    else
    {
        node * p = hidden_temp->nodes;
        while(p->next)
        {
            p = p->next;
        }
        p->next = node_temp;
    }
}

void add_layer_to_graph(layer * hidden_temp, graph * neural_net)
{
    neural_net->no_of_layers++;
    if(neural_net->hidden_layers == NULL)
        neural_net->hidden_layers = hidden_temp;
    else
    {
        layer * p = neural_net->hidden_layers;
        while(p->next)
        {
            p = p->next;
        }
        p->next = hidden_temp;
    }
}

void mat_mul(node * temp_node, float * in, float * out, int node_num)
{
    float local_field = 0;
    for(int i=0;i<temp_node->weight.size();i++)
    {
        local_field += temp_node->weight[i]*in[i];
    }
    out[node_num] = local_field + temp_node->bias;
}

float sigmoid(float num)
{
    return 1/(1+exp(-num));
}

int main()
{
    int n_input, n_hidden, n_output;
    int i, j, k;
    graph * neural_net = new graph;
    neural_net->no_of_layers = 0;
    neural_net->hidden_layers = NULL;
    neural_net->output_layer = NULL;
    cout<<"\nEnter number of input nodes:";
    cin>>n_input;
    no_of_prev_nodes = n_input;
    cout<<"\nEnter number of hidden layers:";
    cin>>n_hidden;
    for(i=0;i<n_hidden;i++)
    {
        int n;
        cout<<"\nHidden Layer "<<i+1<<endl;
        cout<<"\nEnter number of nodes:";
        cin>>n;
        layer * hidden_temp = create_hidden_layer(n);
        for(j=0;j<n;j++)
        {
            float b;
            vector<float> w(no_of_prev_nodes);
            cout<<"\nNode "<<j+1<<endl;
            cout<<"\nEnter the bias:";
            cin>>b;
            cout<<"\nEnter the weights associated to the previous layer:";
            for(k=0;k<no_of_prev_nodes;k++)
            {
                cin>>w[k];
            }
            node * node_temp = create_node(b, w);
            add_node_to_layer(node_temp, hidden_temp);
        }
        no_of_prev_nodes = n;
        add_layer_to_graph(hidden_temp, neural_net);
    }
    cout<<"\nEnter number of output nodes:";
    cin>>n_output;
    neural_net->output_layer = create_hidden_layer(n_output);
    for(i=0;i<n_output;i++)
    {
        float b;
        vector<float> w(no_of_prev_nodes);
        cout<<"\nNode "<<i+1<<endl;
        cout<<"\nEnter the bias:";
        cin>>b;
        cout<<"\nEnter the weights associated to the previous layer:";
        for(k=0;k<no_of_prev_nodes;k++)
            cin>>w[k];
        node * node_temp = create_node(b, w);
        add_node_to_layer(node_temp, neural_net->output_layer);
    }
    add_layer_to_graph(neural_net->output_layer, neural_net);
    cout<<"\nGraph has been built successfully!!\n";
    float in[10], out[10];
    cout<<"\nEnter input to network:";
    for(i=0;i<n_input;i++)
        cin>>in[i];

    layer * temp_layer = neural_net->hidden_layers;
    while(temp_layer)
    {
        node * temp_node = temp_layer->nodes;
        int node_num = 0;
        while(temp_node)
        {
              mat_mul(temp_node, in, out, node_num);
              node_num++;
              temp_node = temp_node->next;
        }
        for(j=0;j<temp_layer->no_of_nodes;j++)
            in[j] = sigmoid(out[j]);
        temp_layer = temp_layer->next;
    }

    for(j=0;j<neural_net->output_layer->no_of_nodes;j++)
        cout<<in[j]<<" ";

    cout<<endl;
    return 0;
}
