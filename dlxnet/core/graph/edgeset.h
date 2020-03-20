#ifndef DLXNET_CORE_GRAPH_EDGESET_H_
#define DLXNET_CORE_GRAPH_EDGESET_H_
#include <stddef.h>
#include "dlxnet/core/platform/macros.h"


namespace dlxnet{
    class Edge;
    // use very little memory for small set
    class EdgeSet{
        public:
            typedef const Edge* value_type;
            typedef const Edge* key_type;
            class const_iterator;
            typedef size_t size_type;
            typedef const_iterator iterator;
            EdgeSet();
            ~EdgeSet();

            bool empty()const;
            size_type size()const;

            std::pair<iterator, bool> insert(value_type value);
            value_type erase(key_type);

            const_iterator begin()const;
            const_iterator end()const;
        private:
            static constexpr int kInline= 64/sizeof(void*);
            const void* ptrs_[kInline];
            TF_DISALLOW_COPY_AND_ASSIGN(EdgeSet);
    };


    class EdgeSet::const_iterator{
        public:
            typedef typename EdgeSet::value_type value_type;
            const_iterator(){}
            const_iterator& operator++();
            bool operator!=(const const_iterator& other)const{
                return !(*this==other);
            }
            bool operator==(const const_iterator& other)const;
            const value_type* operator->()const;
            value_type operator*()const;
    };

    inline EdgeSet::EdgeSet(){
        for(int i=0;i<kInline;i++){
            ptrs_[i] = nullptr;
        }
    }

    inline bool EdgeSet::empty() const { return size() == 0; }

    inline EdgeSet::size_type EdgeSet::size()const{
        return 0;
    }


}


#endif
