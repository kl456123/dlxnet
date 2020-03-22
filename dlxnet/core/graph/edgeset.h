#ifndef DLXNET_CORE_GRAPH_EDGESET_H_
#define DLXNET_CORE_GRAPH_EDGESET_H_
#include <stddef.h>
#include <utility>
#include "dlxnet/core/platform/macros.h"
#include "dlxnet/core/lib/gtl/flatset.h"


namespace dlxnet{
    class Edge;
    // use very little memory for small set
    // ptrs_[0]==this and ptrs_[1]=s when it large than 64
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
            void clear();
            size_type size()const;

            std::pair<iterator, bool> insert(value_type value);
            EdgeSet::size_type erase(key_type key);

            const_iterator begin()const;
            const_iterator end()const;
        private:
            gtl::FlatSet<const Edge*>* get_set()const{
                if(ptrs_[0]==this){
                    return static_cast<gtl::FlatSet<const Edge*>*>(
                            const_cast<void*>(ptrs_[1]));
                }else{
                    // no use set
                    return nullptr;
                }
            }
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
        private:
            friend class EdgeSet;
            void const* const* array_iter_=nullptr;
            typename gtl::FlatSet<const Edge*>::const_iterator tree_iter_;
    };

    inline EdgeSet::EdgeSet(){
        for(int i=0;i<kInline;i++){
            ptrs_[i] = nullptr;
        }
    }

    inline EdgeSet::~EdgeSet(){
        delete get_set();
    }

    inline bool EdgeSet::empty() const { return size() == 0; }

    inline EdgeSet::size_type EdgeSet::size()const{
        auto s = get_set();
        if (s) {
            return s->size();
        } else {
            size_t result = 0;
            for (int i = 0; i < kInline; i++) {
                if (ptrs_[i]) result++;
            }
            return result;
        }
    }

    inline EdgeSet::const_iterator EdgeSet::begin()const{
        const_iterator ci;
        // ci.Init(this);
        auto s = get_set();
        if (s) {
            ci.tree_iter_ = s->begin();
        } else {
            ci.array_iter_ = &ptrs_[0];
        }
        return ci;
    }
    inline EdgeSet::const_iterator EdgeSet::end() const {
        const_iterator ci;
        // ci.Init(this);
        auto s = get_set();
        if (s) {
            ci.tree_iter_ = s->end();
        } else {
            ci.array_iter_ = &ptrs_[size()];
        }
        return ci;
    }
    inline void EdgeSet::clear() {
        // RegisterMutation();
        delete get_set();
        for (int i = 0; i < kInline; i++) {
            ptrs_[i] = nullptr;
        }
    }

    inline EdgeSet::const_iterator& EdgeSet::const_iterator::operator++() {
        if (array_iter_ != nullptr) {
            ++array_iter_;
        } else {
            ++tree_iter_;
        }
        return *this;
    }


    // gcc's set and multiset always use const_iterator since it will otherwise
    // allow modification of keys.
    inline const EdgeSet::const_iterator::value_type* EdgeSet::const_iterator::
        operator->() const {
            // CheckNoMutations();
            if (array_iter_ != nullptr) {
                return reinterpret_cast<const value_type*>(array_iter_);
            } else {
                return tree_iter_.operator->();
            }
        }

    // gcc's set and multiset always use const_iterator since it will otherwise
    // allow modification of keys.
    inline EdgeSet::const_iterator::value_type EdgeSet::const_iterator::operator*()
        const {
            // CheckNoMutations();
            if (array_iter_ != nullptr) {
                return static_cast<value_type>(*array_iter_);
            } else {
                return *tree_iter_;
            }
        }

    inline bool EdgeSet::const_iterator::operator==(
            const const_iterator& other) const {
        DCHECK((array_iter_ == nullptr) == (other.array_iter_ == nullptr))
            << "Iterators being compared must be from same set that has not "
            << "been modified since the iterator was constructed";
        // CheckNoMutations();
        if (array_iter_ != nullptr) {
            return array_iter_ == other.array_iter_;
        } else {
            return other.array_iter_ == nullptr && tree_iter_ == other.tree_iter_;
        }
    }


}


#endif
