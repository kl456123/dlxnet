#ifndef DLXNET_CORE_LIB_GTL_ITERATOR_RANGE_H_
#define DLXNET_CORE_LIB_GTL_ITERATOR_RANGE_H_
#include <utility>

namespace dlxnet{
    namespace gtl{
        // simple iterator range version
        template <typename IteratorT>
            class iterator_range{
                public:
                    iterator_range() : begin_iterator_(), end_iterator_() {}
                    iterator_range(IteratorT x, IteratorT y)
                        :begin_iterator_(x), end_iterator_(y){}
                    IteratorT begin(){return begin_iterator_;}
                    IteratorT end(){return end_iterator_;}

                private:
                    IteratorT begin_iterator_;
                    IteratorT end_iterator_;
            };

        // a Convenience method to make iterator range
        template<typename T>
            iterator_range<T>  make_range(T x, T y){
                return iterator_range<T>(std::move(x),std::move(y)); }
    }
}


#endif
