#ifndef DLXNET_CORE_GRAPH_GRAPH_H_
#define DLXNET_CORE_GRAPH_GRAPH_H_
#include <vector>

namespace dlxnet{
    class Node{
    };

    class Edge{
    };
    class Graph{
        public:
            Graph();
            void AddEdge();
            void AddNode();
        private:
            std::vector<Edge*> edges;
    };

}

#endif
