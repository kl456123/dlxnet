#include "dlxnet/core/framework/kernel_def_util.h"
#include "dlxnet/core/framework/attr_value_util.h"


namespace dlxnet{
    namespace {
        // Helper for KernelAttrsMatch().
        bool InTypeList(DataType dt, const AttrValue& type_list) {
            for (int in_list : type_list.list().type()) {
                if (dt == in_list) return true;
            }
            return false;
        }
    }//namespace
    Status KernelAttrsMatch(const KernelDef& kernel_def, AttrSlice attrs,
            bool* match) {
        *match = false;
        for(const auto& constraint: kernel_def.constraint()){
            auto constraint_value_case = AttrValue::VALUE_NOT_SET;
            int value_type_num = 0;
            if (constraint.allowed_values().list().type_size() > 0) {
                constraint_value_case = AttrValue::kType;
                value_type_num++;
            }
            if (constraint.allowed_values().list().s_size() > 0) {
                constraint_value_case = AttrValue::kS;
                value_type_num++;
            }
            if (constraint.allowed_values().list().i_size() > 0) {
                constraint_value_case = AttrValue::kI;
                value_type_num++;
            }
            if (constraint.allowed_values().list().b_size() > 0) {
                constraint_value_case = AttrValue::kB;
                value_type_num++;
            }
            if (value_type_num == 0) {
                return errors::Unimplemented(
                        "KernelDef '", kernel_def.ShortDebugString(),
                        " has constraint on attr '", constraint.name(),
                        "' with unsupported type: ",
                        SummarizeAttrValue(constraint.allowed_values()));
            }
            if (value_type_num > 1) {
                return errors::InvalidArgument(
                        "KernelDef '", kernel_def.ShortDebugString(),
                        " has constraint on attr '", constraint.name(),
                        "' with more than one value type: ",
                        SummarizeAttrValue(constraint.allowed_values()));
            }

            const AttrValue* attr_value = attrs.Find(constraint.name());
            if (attr_value == nullptr) {
                return errors::InvalidArgument(
                        "OpKernel '", kernel_def.op(), "' has constraint on attr '",
                        constraint.name(), "' not in NodeDef '",
                        "', KernelDef: '", kernel_def.ShortDebugString(), "'");
            }

#define RETURN_IF_ATTR_NOT_FOUND(n, oneof_case, type_str)          \
            do {                                                             \
                if (constraint_value_case == AttrValue::oneof_case) {          \
                    Status s = AttrValueHasType(*attr_value, type_str);          \
                    if (!s.ok()) {                                               \
                        return errors::InvalidArgument(                            \
                                "KernelDef '", kernel_def.ShortDebugString(),          \
                                "' has constraint on attr '", constraint.name(),       \
                                "' that has value '", SummarizeAttrValue(*attr_value), \
                                "' that does not have the same type in NodeDef "       \
                                "'",                                                   \
                                "'");                           \
                    }                                                            \
                    bool found = false;                                          \
                    for (auto& value : constraint.allowed_values().list().n()) { \
                        if (value == attr_value->n()) {                            \
                            found = true;                                            \
                            break;                                                   \
                        }                                                          \
                    }                                                            \
                    if (!found) {                                                \
                        return Status::OK();                                       \
                    }                                                            \
                }                                                              \
            } while (false)

            RETURN_IF_ATTR_NOT_FOUND(s, kS, "string");
            RETURN_IF_ATTR_NOT_FOUND(i, kI, "int");
            RETURN_IF_ATTR_NOT_FOUND(b, kB, "bool");

#undef RETURN_IF_ATTR_NOT_FOUND

            // handle type case
            if (constraint_value_case != AttrValue::kType) {
                continue;
            }

            if (attr_value->type() != DT_INVALID) {
                if (!InTypeList(attr_value->type(), constraint.allowed_values())) {
                    return Status::OK();
                }
            }




        }
        *match = true;
        return Status::OK();
    }
}//namespace dlxnet
