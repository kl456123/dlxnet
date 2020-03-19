#ifndef DLXNET_CORE_GRAPH_EDGESET_H_
#define DLXNET_CORE_GRAPH_EDGESET_H_
#include <stddef.h>
#include "dlxnet/core/platform/macros.h"


namespace dlxnet{
    // use very little memory for small set
    class EdgeSet{
        public:
            class const_iterator;
            typedef size_t size_type;
            EdgeSet();
            ~EdgeSet();

            bool empty()const;
            size_type size()const;
            const_iterator begin()const;
            const_iterator end()const;
        private:
            static constexpr int kInline= 64/sizeof(void*);
            const void* ptrs_[kInline];
            TF_DISALLOW_COPY_AND_ASSIGN(EdgeSet);
    };


    class EdgeSet::const_iterator{
        public:
            const_iterator(){}
            const_iterator& operator++();
    };

    inline EdgeSet::EdgeSet(){
        for(int i=0;i<kInline;i++){
            ptrs_[i] = nullptr;
        }
    }
    inline bool EdgeSet::empty() const { return size() == 0; }

    EdgeSet::size_type EdgeSet::size()const{
        return 0;
    }


}


#endif
