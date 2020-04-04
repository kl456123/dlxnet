#include "dlxnet/core/lib/strings/proto_text_util.h"
#include "absl/strings/escaping.h"
#include "dlxnet/core/platform/protobuf.h"

namespace dlxnet{
    namespace internal{
        bool ProtoParseFromScanner(
                ::dlxnet::strings::Scanner* scanner,
                ::dlxnet::AttrValue* msg){
            std::vector<bool> has_seen(10, false);
            while(true){
                ProtoSpaceAndComments(scanner);
                if(scanner->empty()){return true;}
                scanner->RestartCapture()
                    .Many(Scanner::LETTER_DIGIT_UNDERSCORE)
                    .StopCapture();
                StringPiece identifier;
                if(!scanner->GetResult(nullptr, &identifier))return false;
                bool parsed_colon = false;
                ProtoSpaceAndComments(scanner);
                if (scanner->Peek() == ':') {
                    parsed_colon = true;
                    scanner->One(Scanner::ALL);
                    ProtoSpaceAndComments(scanner);
                }

                if (identifier == "s") {
                    if (msg->value_case() != 0) return false;
                    if (has_seen[0]) return false;
                    has_seen[0] = true;
                    string str_value;
                    if (!parsed_colon || !::dlxnet::strings::ProtoParseStringLiteralFromScanner(
                                scanner, &str_value)) return false;
                    SetProtobufStringSwapAllowed(&str_value, msg->mutable_s());
                }
                else if (identifier == "i") {
                    if (msg->value_case() != 0) return false;
                    if (has_seen[1]) return false;
                    has_seen[1] = true;
                    int64 value;
                    if (!parsed_colon || !::dlxnet::strings::ProtoParseNumericFromScanner(scanner, &value)) return false;
                    msg->set_i(value);
                }
                else if (identifier == "f") {
                    if (msg->value_case() != 0) return false;
                    if (has_seen[2]) return false;
                    has_seen[2] = true;
                    float value;
                    if (!parsed_colon || !::dlxnet::strings::ProtoParseNumericFromScanner(scanner, &value)) return false;
                    msg->set_f(value);
                }
                else if (identifier == "b") {
                    if (msg->value_case() != 0) return false;
                    if (has_seen[3]) return false;
                    has_seen[3] = true;
                    bool value;
                    if (!parsed_colon || !::dlxnet::strings::ProtoParseBoolFromScanner(scanner, &value)) return false;
                    msg->set_b(value);
                }
                else if (identifier == "type") {
                    if (msg->value_case() != 0) return false;
                    if (has_seen[4]) return false;
                    has_seen[4] = true;
                    StringPiece value;
                    if (!parsed_colon || !scanner->RestartCapture().Many(Scanner::LETTER_DIGIT_DASH_UNDERSCORE).GetResult(nullptr, &value)) return false;
                    if (value == "DT_INVALID") {
                        msg->set_type(::dlxnet::DT_INVALID);
                    } else if (value == "DT_FLOAT") {
                        msg->set_type(::dlxnet::DT_FLOAT);
                    } else if (value == "DT_DOUBLE") {
                        msg->set_type(::dlxnet::DT_DOUBLE);
                    } else if (value == "DT_INT32") {
                        msg->set_type(::dlxnet::DT_INT32);
                    } else if (value == "DT_UINT8") {
                        msg->set_type(::dlxnet::DT_UINT8);
                    } else if (value == "DT_INT16") {
                        msg->set_type(::dlxnet::DT_INT16);
                    } else if (value == "DT_INT8") {
                        msg->set_type(::dlxnet::DT_INT8);
                    } else if (value == "DT_STRING") {
                        msg->set_type(::dlxnet::DT_STRING);
                    } else if (value == "DT_COMPLEX64") {
                        msg->set_type(::dlxnet::DT_COMPLEX64);
                    } else if (value == "DT_INT64") {
                        msg->set_type(::dlxnet::DT_INT64);
                    } else if (value == "DT_BOOL") {
                        msg->set_type(::dlxnet::DT_BOOL);
                    } else if (value == "DT_QINT8") {
                        msg->set_type(::dlxnet::DT_QINT8);
                    } else if (value == "DT_QUINT8") {
                        msg->set_type(::dlxnet::DT_QUINT8);
                    } else if (value == "DT_QINT32") {
                        msg->set_type(::dlxnet::DT_QINT32);
                    } else if (value == "DT_BFLOAT16") {
                        msg->set_type(::dlxnet::DT_BFLOAT16);
                    } else if (value == "DT_QINT16") {
                        msg->set_type(::dlxnet::DT_QINT16);
                    } else if (value == "DT_QUINT16") {
                        msg->set_type(::dlxnet::DT_QUINT16);
                    } else if (value == "DT_UINT16") {
                        msg->set_type(::dlxnet::DT_UINT16);
                    } else if (value == "DT_COMPLEX128") {
                        msg->set_type(::dlxnet::DT_COMPLEX128);
                    } else if (value == "DT_HALF") {
                        msg->set_type(::dlxnet::DT_HALF);
                    } else if (value == "DT_RESOURCE") {
                        msg->set_type(::dlxnet::DT_RESOURCE);
                    } else if (value == "DT_VARIANT") {
                        msg->set_type(::dlxnet::DT_VARIANT);
                    } else if (value == "DT_UINT32") {
                        msg->set_type(::dlxnet::DT_UINT32);
                    } else if (value == "DT_UINT64") {
                        msg->set_type(::dlxnet::DT_UINT64);
                    } else if (value == "DT_FLOAT_REF") {
                        msg->set_type(::dlxnet::DT_FLOAT_REF);
                    } else if (value == "DT_DOUBLE_REF") {
                        msg->set_type(::dlxnet::DT_DOUBLE_REF);
                    } else if (value == "DT_INT32_REF") {
                        msg->set_type(::dlxnet::DT_INT32_REF);
                    } else if (value == "DT_UINT8_REF") {
                        msg->set_type(::dlxnet::DT_UINT8_REF);
                    } else if (value == "DT_INT16_REF") {
                        msg->set_type(::dlxnet::DT_INT16_REF);
                    } else if (value == "DT_INT8_REF") {
                        msg->set_type(::dlxnet::DT_INT8_REF);
                    } else if (value == "DT_STRING_REF") {
                        msg->set_type(::dlxnet::DT_STRING_REF);
                    } else if (value == "DT_COMPLEX64_REF") {
                        msg->set_type(::dlxnet::DT_COMPLEX64_REF);
                    } else if (value == "DT_INT64_REF") {
                        msg->set_type(::dlxnet::DT_INT64_REF);
                    } else if (value == "DT_BOOL_REF") {
                        msg->set_type(::dlxnet::DT_BOOL_REF);
                    } else if (value == "DT_QINT8_REF") {
                        msg->set_type(::dlxnet::DT_QINT8_REF);
                    } else if (value == "DT_QUINT8_REF") {
                        msg->set_type(::dlxnet::DT_QUINT8_REF);
                    } else if (value == "DT_QINT32_REF") {
                        msg->set_type(::dlxnet::DT_QINT32_REF);
                    } else if (value == "DT_BFLOAT16_REF") {
                        msg->set_type(::dlxnet::DT_BFLOAT16_REF);
                    } else if (value == "DT_QINT16_REF") {
                        msg->set_type(::dlxnet::DT_QINT16_REF);
                    } else if (value == "DT_QUINT16_REF") {
                        msg->set_type(::dlxnet::DT_QUINT16_REF);
                    } else if (value == "DT_UINT16_REF") {
                        msg->set_type(::dlxnet::DT_UINT16_REF);
                    } else if (value == "DT_COMPLEX128_REF") {
                        msg->set_type(::dlxnet::DT_COMPLEX128_REF);
                    } else if (value == "DT_HALF_REF") {
                        msg->set_type(::dlxnet::DT_HALF_REF);
                    } else if (value == "DT_RESOURCE_REF") {
                        msg->set_type(::dlxnet::DT_RESOURCE_REF);
                    } else if (value == "DT_VARIANT_REF") {
                        msg->set_type(::dlxnet::DT_VARIANT_REF);
                    } else if (value == "DT_UINT32_REF") {
                        msg->set_type(::dlxnet::DT_UINT32_REF);
                    } else if (value == "DT_UINT64_REF") {
                        msg->set_type(::dlxnet::DT_UINT64_REF);
                    } else {
                        int32 int_value;
                        if (strings::SafeStringToNumeric(value, &int_value)) {
                            msg->set_type(static_cast<::dlxnet::DataType>(int_value));
                        } else {
                            return false;
                        }
                    }
                }
            }
        }
    }// namespace internal

    bool ProtoParseFromString(
            const string& s,
            ::dlxnet::AttrValue* msg) {
        msg->Clear();
        Scanner scanner(s);
        if (!internal::ProtoParseFromScanner(&scanner, msg)) return false;
        scanner.Eos();
        return scanner.GetResult();
    }

    namespace strings{
        bool ProtoParseBoolFromScanner(Scanner* scanner, bool* value) {
            StringPiece bool_str;
            if (!scanner->RestartCapture()
                    .Many(Scanner::LETTER_DIGIT)
                    .GetResult(nullptr, &bool_str)) {
                return false;
            }
            ProtoSpaceAndComments(scanner);
            if (bool_str == "false" || bool_str == "False" || bool_str == "0") {
                *value = false;
                return true;
            } else if (bool_str == "true" || bool_str == "True" || bool_str == "1") {
                *value = true;
                return true;
            } else {
                return false;
            }
        }

        bool ProtoParseStringLiteralFromScanner(Scanner* scanner, string* value) {
            const char quote = scanner->Peek();
            if (quote != '\'' && quote != '"') return false;

            StringPiece value_sp;
            if (!scanner->One(Scanner::ALL)
                    .RestartCapture()
                    .ScanEscapedUntil(quote)
                    .StopCapture()
                    .One(Scanner::ALL)
                    .GetResult(nullptr, &value_sp)) {
                return false;
            }
            ProtoSpaceAndComments(scanner);
            return absl::CUnescape(value_sp, value, nullptr /* error */);
        }
    }
}
