#ifndef DLXNET_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
#define DLXNET_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_

#include "absl/strings/str_cat.h"
#include "dlxnet/core/platform/scanner.h"
#include "dlxnet/core/platform/numbers.h"
#include "dlxnet/core/framework/attr_value.pb.h"


using ::dlxnet::strings::Scanner;

namespace dlxnet{
    namespace internal{
        bool ProtoParseFromScanner(
                ::dlxnet::strings::Scanner* scanner,
                ::dlxnet::AttrValue* msg);
    }
    namespace strings{
        // some types parser
        inline void ProtoSpaceAndComments(Scanner* scanner){
            for(;;){
                scanner->AnySpace();
                if(scanner->Peek()!='#')return;
                while(scanner->Peek('\n')!='\n')scanner->One(Scanner::ALL);
            }
        }

        // Parse the next numeric value from <scanner>, returning false if parsing
        // failed.
        template <typename T>
            bool ProtoParseNumericFromScanner(Scanner* scanner, T* value) {
                StringPiece numeric_str;
                scanner->RestartCapture();
                if (!scanner->Many(Scanner::LETTER_DIGIT_DOT_PLUS_MINUS)
                        .GetResult(nullptr, &numeric_str)) {
                    return false;
                }

                // Special case to disallow multiple leading zeroes, to match proto parsing.
                int leading_zero = 0;
                for (size_t i = 0; i < numeric_str.size(); ++i) {
                    const char ch = numeric_str[i];
                    if (ch == '0') {
                        if (++leading_zero > 1) return false;
                    } else if (ch != '-') {
                        break;
                    }
                }

                ProtoSpaceAndComments(scanner);
                return SafeStringToNumeric<T>(numeric_str, value);
            }

        // Parse the next boolean value from <scanner>, returning false if parsing
        // failed.
        bool ProtoParseBoolFromScanner(Scanner* scanner, bool* value);

        // Parse the next string literal from <scanner>, returning false if parsing
        // failed.
        bool ProtoParseStringLiteralFromScanner(Scanner* scanner, string* value);
    }

    bool ProtoParseFromString(const string& s,
            ::dlxnet::AttrValue* msg);
}


#endif
