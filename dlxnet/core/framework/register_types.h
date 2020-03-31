#ifndef DLXNET_CORE_FRAMEWORK_REGISTER_TYPES_H_
#define DLXNET_CORE_FRAMEWORK_REGISTER_TYPES_H_

#define TF_CALL_float(m) m(float)
#define TF_CALL_double(m) m(double)
#define TF_CALL_int32(m) m(::dlxnet::int32)
#define TF_CALL_uint32(m) m(::dlxnet::uint32)
#define TF_CALL_uint8(m) m(::dlxnet::uint8)
#define TF_CALL_int16(m) m(::dlxnet::int16)

#define TF_CALL_int8(m) m(::dlxnet::int8)
#define TF_CALL_string(m) m(::dlxnet::tstring)
#define TF_CALL_tstring(m) m(::dlxnet::tstring)


#endif
