#ifndef DLXNET_CORE_GRAPH_GRAPH_H_
#define DLXNET_CORE_GRAPH_GRAPH_H_
#include <vector>
#include <memory>

#include "dlxnet/core/framework/types.h"
#include "dlxnet/core/lib/gtl/iterator_range.h"
#include "dlxnet/core/lib/status.h"
#include "dlxnet/core/framework/graph.pb.h"
#include "dlxnet/core/framework/node_def_util.h"
#include "dlxnet/core/graph/edgeset.h"
#include "dlxnet/core/framework/op.h"
#include "dlxnet/core/lib/core/arena.h"

namespace dlxnet{
    class Node;
    class Edge;
    class Graph;
    class GraphDef;

    // iterator
    class NodeIter;
    class NeighborIter;

    struct NodeProperties;

    class Node{
        public:
            string DebugString() const;
            const string& name()const;
            int id() const { return id_; }
            void set_name(string name);
            const string& type_string()const;
            // This gives the device the runtime has assigned this node to.  If
            // you want the device the user requested, use def().device() instead.
            // TODO(josh11b): Validate that the assigned_device, if not empty:
            // fully specifies a device, and satisfies def().device().
            // TODO(josh11b): Move assigned_device_name outside of Node into a
            // NodeId->DeviceName map.
            const string& assigned_device_name() const;
            void set_assigned_device_name(const string& device_name);
            void set_assigned_device_name_index(int index);
            bool has_assigned_device_name() const {
                return assigned_device_name_index_ > 0;
            }
            int assigned_device_name_index() const { return assigned_device_name_index_; }

            // def() provides the NodeDef the user supplied, but the specifics
            // of this Node may have changed due to placement, optimization, etc.
            // In particular:
            // * def().name() will match name();
            // * def().op() will match type_string() and op_def().name();
            // * def().input() is not reliable, use "in_edges()" below instead;
            // * def().device() is the "user's requested device" and may not match
            //   the actual assigned device, see assigned_device_name() below;
            // * def().attr() is authoritative.
            // TODO(irving): Replace with NodeInfo.
            const NodeDef& def() const;
            const OpDef& op_def() const;

            // input and output types
            int32 num_inputs() const;
            DataType input_type(int32 i) const;
            const DataTypeVector& input_types() const;

            int32 num_outputs() const;
            DataType output_type(int32 o) const;
            const DataTypeVector& output_types() const;

            // Read only access to attributes
            AttrSlice attrs() const;

            // Get the neighboring nodes via edges either in or out of this node.  This
            // includes control edges.
            gtl::iterator_range<NeighborIter> in_nodes() const;
            gtl::iterator_range<NeighborIter> out_nodes() const;
            const EdgeSet& in_edges() const { return in_edges_; }
            const EdgeSet& out_edges() const { return out_edges_; }


            // attr helper functions
            template <typename T>
                void AddAttr(const string& name, const T& val) {
                    // SetAttrValue(val, AddAttrHelper(name));
                    UpdateProperties();
                }

            void ClearAttr(const string& name);

            // Returns into '*e' the edge connecting to the 'idx' input of this Node.
            Status input_edge(int idx, const Edge** e) const;

            // Returns into '*edges' the input data edges of this Node, indexed by input
            // number. Does not return control edges.
            Status input_edges(std::vector<const Edge*>* edges) const;

            // Returns into '*n' the node that has an output connected to the
            // 'idx' input of this Node.
            Status input_node(int idx, const Node** n) const;
            Status input_node(int idx, Node** n) const;

            // Node type helpers.
            bool IsSource() const { return id() == 0; }
            bool IsSink() const { return id() == 1; }

        private:
            // Called after an attr has changed. Decides whether we need to update some
            // property of the node (stored in props_).
            void UpdateProperties();

            NodeProperties* properties() const { return props_.get(); }

            void Initialize(int id,  std::shared_ptr<NodeProperties> props);


            enum NodeClass{
                NC_UNINITIALIZED,
                NC_OTHER
            };
            static NodeClass GetNodeClassForOp(const string& ts);

            friend class Graph;
            Node();
            int id_;       // -1 until Initialize() is called
            NodeClass class_;

            EdgeSet in_edges_;
            EdgeSet out_edges_;
            // Index within Graph::device_names_ of the name of device assigned
            // to perform this computation.
            int assigned_device_name_index_;

            // A back-pointer to the Graph that owns this node.  Currently, this exists
            // solely to allow Node::[set_]assigned_device_name() to work. However, if all
            // callers of Node::[set_]assigned_device_name() are modified to use the
            // equivalent methods defined directly on Graph, then we can remove this
            // field and reclaim that memory.
            Graph* graph_;

            // NOTE(skyewm): inheriting from core::RefCounted may have a slight
            // performance benefit over using shared_ptr, at the cost of manual ref
            // counting
            std::shared_ptr<NodeProperties> props_;

            TF_DISALLOW_COPY_AND_ASSIGN(Node);


    };

    class Edge{
        public:
            Node* src() const { return src_; }
            Node* dst() const { return dst_; }
            int id() const { return id_; }

            // Return the index of the source output that produces the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int src_output() const { return src_output_; }

            // Return the index of the destination input that consumes the data
            // carried by this edge.  The special value kControlSlot is used
            // for control dependencies.
            int dst_input() const { return dst_input_; }

            string DebugString() const;

        private:
            Edge() {}
            friend class Graph;
            Node* src_;
            Node* dst_;
            int id_;
            int src_output_;
            int dst_input_;
    };
    // Allows for iteration of the edges of a Graph, by iterating the underlying
    // Graph.edges_ vector while skipping over null entries.
    class GraphEdgesIterable{
        private:
            const std::vector<Edge*>& edges_;
        public:
            explicit GraphEdgesIterable(const std::vector<Edge*>& edges)
                : edges_(edges) {}
            typedef Edge* value_type;
            class const_iterator{
                private:
                    // iterator of vector
                    std::vector<value_type>::const_iterator iter_;
                    std::vector<value_type>::const_iterator end_;
                    // skip empty
                    void skip_empty() {
                        while (iter_ != end_ && *iter_ == nullptr) {
                            ++iter_;
                        }
                    }
                public:
                    const_iterator(std::vector<value_type>::const_iterator iter,
                            std::vector<value_type>::const_iterator end)
                        : iter_(iter), end_(end) {
                            skip_empty();
                        }
                    const_iterator& operator++(){
                        ++iter_;
                        skip_empty();
                        return *this;
                    }
                    bool operator==(const const_iterator& other) const {
                        return iter_ == other.iter_;
                    }

                    bool operator!=(const const_iterator& other) const {
                        return iter_ != other.iter_;
                    }

                    value_type operator*() { return *iter_; }

            };
            const_iterator begin() {
                return const_iterator(edges_.begin(), edges_.end());
            }
            const_iterator end() { return const_iterator(edges_.end(), edges_.end()); }

    };

    class Graph{
        public:
            // Constructs a graph with a single SOURCE (always id kSourceId) and a
            // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
            //
            // The graph can hold ops found in the registry. `ops`s lifetime must be at
            // least that of the constructed graph's.
            explicit Graph(const OpRegistryInterface* ops);
            ~Graph();
            // Adds a new node to this graph, and returns it. Infers the Op and
            // input/output types for the node. *this owns the returned instance.
            // Returns nullptr and sets *status on error.
            Node* AddNode(NodeDef node_def, Status* status);

            // Adds an edge that connects the xth output of `source` to the yth input of
            // `dest` and returns it. Does not update dest's NodeDef.
            const Edge* AddEdge(Node* source, int x, Node* dest, int y);

            // The number of live nodes in the graph.
            //
            // Because nodes can be removed from the graph, num_nodes() is often
            // smaller than num_node_ids(). If one needs to create an array of
            // nodes indexed by node ids, num_node_ids() should be used as the
            // array's size.
            int num_nodes() const { return num_nodes_; }

            // Returns one more than the maximum id assigned to any node.
            int num_node_ids() const { return nodes_.size(); }

            // The number of live nodes in the graph, excluding the Source and Sink nodes.
            int num_op_nodes() const {
                DCHECK_GE(num_nodes_, 2);
                return num_nodes_ - 2;
            }
            // Serialize to a GraphDef.
            void ToGraphDef(GraphDef* graph_def) const;

            // Generate new node name with the specified prefix that is unique
            // across this graph.
            string NewName(StringPiece prefix);

            Node* CopyNode(const Node* node);

            // Access to the list of all nodes.  Example usage:
            //   for (Node* node : graph.nodes()) { ... }
            gtl::iterator_range<NodeIter> nodes() const;

            // Access to the list of all nodes, excluding the Source and Sink nodes.
            gtl::iterator_range<NodeIter> op_nodes() const;

            // Access to the set of all edges.  Example usage:
            //   for (const Edge* e : graph.edges()) { ... }
            GraphEdgesIterable edges() const { return GraphEdgesIterable(edges_); }
            // Returns the node associated with an id, or nullptr if no node
            // with that id (the node with that id was removed and the id has
            // not yet been re-used). *this owns the returned instance.
            // REQUIRES: 0 <= id < num_node_ids().
            Node* FindNodeId(int id) const { return nodes_[id]; }

            // The pre-defined nodes.
            enum { kSourceId = 0, kSinkId = 1 };
            Node* source_node() const { return FindNodeId(kSourceId); }
            Node* sink_node() const { return FindNodeId(kSinkId); }
            const OpRegistryInterface* op_registry() const { return ops_; }
            void CheckDeviceNameIndex(int index) {
                DCHECK_GE(index, 0);
                DCHECK_LT(index, static_cast<int>(device_names_.size()));
            }
            // Builds a node name to node pointer index for all nodes in the graph.
            std::unordered_map<string, Node*> BuildNodeNameIndex() const;

            // device_name for node
            const string& get_assigned_device_name(const Node& node) const {
                return device_names_[node.assigned_device_name_index()];
            }

            void set_assigned_device_name_index(Node* node, int device_name_index) {
                CheckDeviceNameIndex(device_name_index);
                node->assigned_device_name_index_ = device_name_index;
            }

            void set_assigned_device_name(Node* node, const string& device_name) {
                // TODO(add device name map for efficiency)
                node->assigned_device_name_index_ = InternDeviceName(device_name);
            }


            int InternDeviceName(const string& device_name);

        private:
            // If cost_node is non-null, then cost accounting (in CostModel)
            // will be associated with that node rather than the new one being
            // created.
            //
            // Ownership of the returned Node is not transferred to caller.
            Node* AllocateNode(std::shared_ptr<NodeProperties> props);
            void ReleaseNode(Node* node);

            std::vector<Edge*> edges_;
            // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
            // the node with that id was removed from the graph.
            std::vector<Node*> nodes_;
            // The number of entries in edges_ that are not nullptr.
            int num_edges_ = 0;
            // For generating unique names.
            int name_counter_ = 0;

            // Allocated but free nodes and edges.
            std::vector<Node*> free_nodes_;
            std::vector<Edge*> free_edges_;

            // Number of nodes alive.
            int64 num_nodes_ = 0;

            // Registry of all known ops, including functions.
            OpRegistry* ops_;

            // In most graphs, the number of unique values used for the
            // Node::assigned_device_name() property is quite small.  If the graph is
            // large, then this duplication of values can consume a significant amount of
            // memory.  Instead, we represent the same information using an interning
            // table, which consists of a vector of unique strings (device_names_), as
            // well a map (device_names_map_) from unique strings to indices within the
            // unique string table.
            //
            // The InternDeviceName() method handles adding a new entry into the table,
            // or locating the index of an existing entry.
            //
            // The fact that Node::assigned_device_name() is implemented using an
            // interning table is intentionally public.  This allows algorithms that
            // frequently access this field to do so efficiently, especially for the case
            // where the assigned_device_name of one Node is copied directly from that
            // of another Node.

            // A table of the unique assigned device names.  Indices do NOT correspond
            // to node IDs.  Index 0 is always the empty string.
            std::vector<string> device_names_;

            // Maps unique device names to indices within device_names_[i].
            std::unordered_map<string, int> device_names_map_;

            // Allocator which will give us good locality.
            core::Arena arena_;
            TF_DISALLOW_COPY_AND_ASSIGN(Graph);
    };
    inline bool IsSource(const Node* node) { return node->IsSource(); }
    inline bool IsSink(const Node* node) { return node->IsSink(); }


    class NodeIter : public std::iterator<std::forward_iterator_tag, Node, std::ptrdiff_t,
    /*Pointer*/ Node*, /*Reference*/ Node*>{
        public:
            NodeIter(const Graph* graph, int id);
            bool operator==(const NodeIter& rhs) const;
            bool operator!=(const NodeIter& rhs) const;
            void operator++();
            reference operator*() const;
            pointer operator->() const;

        private:
            // Invariant: id_ == graph_->num_node_ids() || graph_->FindId(id_) != nullptr
            const Graph* graph_;
            int id_;
    };

    // Iterator for stepping through the neighbors of a node.
    class NeighborIter
        : public std::iterator<std::forward_iterator_tag, Node, std::ptrdiff_t,
        /*Pointer*/ Node*, /*Reference*/ Node*> {
            public:
                NeighborIter(EdgeSet::const_iterator iter, bool incoming);
                bool operator==(const NeighborIter& rhs) const;
                bool operator!=(const NeighborIter& rhs) const;
                void operator++();
                reference operator*() const;
                pointer operator->() const;

            private:
                EdgeSet::const_iterator iter_;
                bool incoming_;
        };

    // implement for iterator

    inline NodeIter::NodeIter(const Graph* graph, int id)
        : graph_(graph), id_(id) {}

    inline void NodeIter::operator++() {
        // skip empty(nullptr)
        while (1) {
            DCHECK_LE(id_, graph_->num_node_ids());
            ++id_;
            if (id_ >= graph_->num_node_ids() || graph_->FindNodeId(id_) != nullptr) {
                return;
            }
        }
    }

    inline Node* NodeIter::operator*() const { return graph_->FindNodeId(id_); }

    inline Node* NodeIter::operator->() const { return graph_->FindNodeId(id_); }

    inline bool NodeIter::operator==(const NodeIter& rhs) const {
        DCHECK(graph_ == rhs.graph_);
        return id_ == rhs.id_;
    }

    inline bool NodeIter::operator!=(const NodeIter& rhs) const {
        return !(*this == rhs);
    }
    inline NeighborIter::NeighborIter(EdgeSet::const_iterator iter, bool incoming)
        : iter_(iter), incoming_(incoming) {}
    inline bool NeighborIter::operator==(const NeighborIter& rhs) const {
        return iter_ == rhs.iter_ && incoming_ == rhs.incoming_;
    }

    inline bool NeighborIter::operator!=(const NeighborIter& rhs) const {
        return !(*this == rhs);
    }
    inline void NeighborIter::operator++() { ++iter_; }

    inline Node* NeighborIter::operator*() const {
        const Edge* e = *iter_;
        return incoming_ ? e->src() : e->dst();
    }

    inline Node* NeighborIter::operator->() const {
        const Edge* e = *iter_;
        return incoming_ ? e->src() : e->dst();
    }

    inline gtl::iterator_range<NodeIter> Graph::nodes() const {
        // Note that NodeId 0 is always valid since we don't let the source
        // node be removed from the graph.
        return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
    }

    inline gtl::iterator_range<NodeIter> Graph::op_nodes() const {
        // the same as nodes() due to ignore sink and source node
        return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
    }


    // node set device_name
    inline void Node::set_assigned_device_name_index(int index) {
        graph_->CheckDeviceNameIndex(index);
        assigned_device_name_index_ = index;
    }

    inline void Node::set_assigned_device_name(const string& device_name) {
        graph_->set_assigned_device_name(this, device_name);
    }

    inline const string& Node::assigned_device_name() const {
        return graph_->get_assigned_device_name(*this);
    }
}

#endif
