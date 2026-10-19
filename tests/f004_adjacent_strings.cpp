/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
// F004 standalone driver. Compile with build-tools/minja-f004-adjacent-strings.cmd.
// Header-only Minja plus the nlohmann single include. No GenAI or OVMS link.

#include <minja/chat-template.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

void pass(const std::string & name) {
    std::cout << "PASS " << name << "\n";
}

void fail(const std::string & name, const std::string & detail) {
    std::cerr << "FAIL " << name << ": " << detail << "\n";
    ++g_failures;
}

struct RenderResult {
    bool ok = false;
    std::string output;
    std::string error;
};

RenderResult render_template(const std::string & source, const minja::Options & options, const json & bindings) {
    RenderResult result;
    try {
        auto root = minja::Parser::parse(source, options);
        auto context = minja::Context::make(minja::Value(bindings));
        result.output = root->render(context);
        result.ok = true;
    } catch (const std::exception & e) {
        result.error = e.what();
    }
    return result;
}

void expect_render(const std::string & name, const std::string & source, const std::string & expected) {
    auto result = render_template(source, {}, json::object());
    if (!result.ok) {
        fail(name, "threw: " + result.error);
        return;
    }
    if (result.output != expected) {
        std::ostringstream detail;
        detail << "got " << json(result.output).dump() << " expected " << json(expected).dump();
        fail(name, detail.str());
        return;
    }
    pass(name);
}

void expect_throw_prefix(const std::string & name, const std::string & source, const std::string & expected_prefix) {
    auto result = render_template(source, {}, json::object());
    if (result.ok) {
        fail(name, "rendered " + json(result.output).dump() + " instead of throwing");
        return;
    }
    if (result.error.compare(0, expected_prefix.size(), expected_prefix) != 0) {
        fail(name, "threw: " + result.error);
        return;
    }
    if (result.error.size() > expected_prefix.size()) {
        char next = result.error[expected_prefix.size()];
        if (next != ' ' && next != '\n' && next != '\r') {
            fail(name, "threw: " + result.error);
            return;
        }
    }
    pass(name);
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: f004_adjacent_strings <gemma-chat-template.jinja> <qwen-chat-template.jinja>\n";
        return 2;
    }
    const std::string gemma_path = argv[1];
    const std::string qwen_path = argv[2];

    expect_render("harmless_adjacent_strings", "{{ \"a\" \"b\" }}", "ab");
    expect_render("three_adjacent_strings", "{{ \"a\" \"b\" \"c\" }}", "abc");
    expect_render("mixed_quotes", "{{ \"a\" 'b' \"c\" }}", "abc");
    expect_render("whitespace_inside_preserved", "{{ \"a \" \" b\" }}", "a  b");
    expect_render("newline_between_literals_not_inserted", "{{ \"a\"\n\"b\" }}", "ab");
    expect_render("escaped_newline_preserved", "{{ \"a\\n\" \"b\" }}", "a\nb");
    expect_render("escaped_tab_preserved", "{{ \"a\\t\" \"b\" }}", "a\tb");
    expect_render("backslash_preserved", "{{ \"a\\\\b\" 'c\\\\d' }}", "a\\bc\\d");
    expect_render("escaped_quote_preserved", "{{ \"say \\\"hi\\\"\" ' there' }}", "say \"hi\" there");
    expect_render("implicit_before_plus", "{{ \"a\" \"b\" + \"c\" }}", "abc");
    expect_render("plus_then_implicit", "{{ \"a\" + \"b\" \"c\" }}", "abc");
    expect_render("filter_after_concat", "{{ \"a\" \"b\" | upper }}", "AB");
    expect_render("single_string_unchanged", "{{ \"ab\" }}", "ab");
    expect_render("explicit_plus_unchanged", "{{ \"a\" + \"b\" }}", "ab");
    expect_render("bool_true_unchanged", "{{ true }}", "True");
    expect_render("bool_false_unchanged", "{{ False }}", "False");
    expect_render("none_string_unchanged", "{{ None | string }}", "None");
    expect_render("integer_unchanged", "{{ 12 }}", "12");
    expect_render("addition_unchanged", "{{ 1 + 2 }}", "3");
    expect_render("comma_tuple_not_concat", "{{ (\"a\", \"b\") }}", "['a', 'b']");
    expect_render("adjacent_inside_tuple_differs_from_comma", "{{ (\"a\" \"b\", \"c\" \"d\") }}", "['ab', 'cd']");

    expect_throw_prefix(
        "raise_exception_adjacent_message",
        "{{ raise_exception(\"a\" \"b\") }}",
        "ab");
    expect_throw_prefix(
        "raise_exception_comma_args_stay_separate",
        "{{ raise_exception(\"a\", \"b\") }}",
        "Too many positional params for raise_exception");
    expect_throw_prefix(
        "gemma_shaped_raise_exception",
        "{{ raise_exception(\n"
        "    \"chat_template: tool_calls[].function.arguments must be a \"\n"
        "    \"JSON object (mapping), not a string. Deserialize arguments \"\n"
        "    \"before passing to the template.\"\n"
        ") }}",
        "chat_template: tool_calls[].function.arguments must be a "
        "JSON object (mapping), not a string. Deserialize arguments "
        "before passing to the template.");
    expect_throw_prefix("numeric_adjacency_not_coerced", "{{ 1 2 }}", "Expected closing expression tag");
    expect_throw_prefix("numeric_newline_adjacency_not_coerced", "{{ 1\n2 }}", "Expected closing expression tag");
    expect_throw_prefix("string_then_number_not_coerced", "{{ \"1\" 2 }}", "Expected closing expression tag");
    expect_throw_prefix("number_then_string_not_coerced", "{{ 1 \"2\" }}", "Expected closing expression tag");

    std::string gemma_source;
    try {
        gemma_source = read_file(gemma_path);
    } catch (const std::exception & e) {
        fail("gemma_fixture_readable", e.what());
        std::cout << "failures " << g_failures << "\n";
        return 1;
    }
    if (gemma_source.find("\"chat_template: tool_calls[].function.arguments must be a \"") == std::string::npos) {
        fail("gemma_fixture_is_original_template", "adjacent raise_exception strings missing");
    } else {
        pass("gemma_fixture_is_original_template");
    }

    const std::string gemma_basic =
        "<bos><|turn>user\n"
        "Hi<turn|>\n"
        "<|turn>model\n"
        "probe_basic_7Qw2<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n"
        "<channel|>";
    try {
        minja::chat_template gemma(gemma_source, "<bos>", "<eos>");
        minja::chat_template_inputs inputs;
        inputs.messages = json::array({
            json{{"role", "user"}, {"content", "Hi"}},
            json{{"role", "assistant"}, {"content", "probe_basic_7Qw2"}},
        });
        inputs.add_generation_prompt = true;
        minja::chat_template_options opts;
        opts.apply_polyfills = false;
        auto rendered = gemma.apply(inputs, opts);
        if (rendered != gemma_basic) {
            fail("gemma_basic_render", "got " + json(rendered).dump());
        } else {
            pass("gemma_basic_render");
        }

        inputs.messages = json::array({
            json{{"role", "user"}, {"content", "Hi"}},
            json{
                {"role", "assistant"},
                {"tool_calls", json::array({
                    json{
                        {"id", "call_1"},
                        {"type", "function"},
                        {"function", {
                            {"name", "get_weather"},
                            {"arguments", "{\"city\":\"Oslo\"}"},
                        }},
                    },
                })},
            },
        });
        inputs.add_generation_prompt = false;
        bool threw = false;
        std::string error;
        try {
            gemma.apply(inputs, opts);
        } catch (const std::exception & e) {
            threw = true;
            error = e.what();
        }
        const std::string expected_message =
            "chat_template: tool_calls[].function.arguments must be a "
            "JSON object (mapping), not a string. Deserialize arguments "
            "before passing to the template.";
        const bool starts = error.compare(0, expected_message.size(), expected_message) == 0;
        const bool bounded = !starts || error.size() == expected_message.size() ||
            error[expected_message.size()] == ' ' || error[expected_message.size()] == '\n' ||
            error[expected_message.size()] == '\r';
        if (!threw || !starts || !bounded) {
            fail("gemma_string_arguments_raise", threw ? error : "rendered without raise_exception");
        } else {
            pass("gemma_string_arguments_raise");
        }

        inputs.messages = json::array({
            json{{"role", "user"}, {"content", "Hi"}},
            json{
                {"role", "assistant"},
                {"tool_calls", json::array({
                    json{
                        {"id", "call_1"},
                        {"type", "function"},
                        {"function", {
                            {"name", "get_weather"},
                            {"arguments", json{{"city", "Oslo"}}},
                        }},
                    },
                })},
            },
        });
        try {
            auto tool_rendered = gemma.apply(inputs, opts);
            if (tool_rendered.find("<|tool_call>call:get_weather{") == std::string::npos ||
                tool_rendered.find("city:<|\"|>Oslo<|\"|>") == std::string::npos ||
                tool_rendered.find("}<tool_call|>") == std::string::npos) {
                fail("gemma_object_arguments_render", json(tool_rendered).dump());
            } else {
                pass("gemma_object_arguments_render");
            }
        } catch (const std::exception & e) {
            fail("gemma_object_arguments_render", e.what());
        }
    } catch (const std::exception & e) {
        fail("gemma_template_parse", e.what());
    }

    try {
        auto qwen_source = read_file(qwen_path);
        minja::chat_template qwen(qwen_source, "", "");
        minja::chat_template_inputs inputs;
        inputs.messages = json::array({
            json{{"role", "user"}, {"content", "Hi"}},
        });
        inputs.add_generation_prompt = true;
        minja::chat_template_options opts;
        opts.apply_polyfills = false;
        auto rendered = qwen.apply(inputs, opts);
        const std::string expected = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        if (rendered != expected) {
            fail("qwen_basic_render", "got " + json(rendered).dump());
        } else {
            pass("qwen_basic_render");
        }
    } catch (const std::exception & e) {
        fail("qwen_basic_render", e.what());
    }

    std::cout << "failures " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}
