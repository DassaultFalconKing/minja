/* Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
// Production-shaped regression: chat_template::apply uses its default options.
#include <minja/chat-template.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
void expect(bool value, const std::string & name) {
    std::cout << (value ? "PASS " : "FAIL ") << name << '\n';
    if (!value) ++failures;
}
std::string read(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}
bool contains(const std::string & text, const std::string & needle) {
    return text.find(needle) != std::string::npos;
}
json tool_call(const json & args) {
    return {{"id", "call_weather_1"}, {"type", "function"},
            {"function", {{"name", "get_weather"}, {"arguments", args}}}};
}
minja::chat_template_inputs replay(const json & args) {
    minja::chat_template_inputs in;
    in.messages = json::array({
        {{"role", "user"}, {"content", "Weather?"}},
        {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({tool_call(args)})}},
        // Resolve the name through tool_call_id, as real OpenAI tool responses do.
        {{"role", "tool"}, {"tool_call_id", "call_weather_1"}, {"content", "Weather is sunny"}},
    });
    return in;
}
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: gemma4_default_polyfills <gemma.jinja> <qwen.jinja> [evidence-dir]\n";
        return 2;
    }
    try {
        minja::chat_template gemma(read(argv[1]), "<bos>", "<eos>");
        const auto caps = gemma.original_caps();
        expect(caps.supports_tools, "gemma_supports_tools");
        expect(caps.supports_tool_calls, "gemma_supports_tool_calls");
        expect(caps.requires_object_arguments, "gemma_requires_object_arguments");
        expect(caps.supports_tool_responses, "gemma_supports_tool_responses");
        expect(caps.supports_parallel_tool_calls, "gemma_supports_parallel_tool_calls");

        const json args{{"city", "Oslo"}};
        auto in = replay(args);
        const auto original_messages = in.messages;
        const std::string expected =
            "<bos><|turn>user\nWeather?<turn|>\n<|turn>model\n"
            "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>}<tool_call|>"
            "<|tool_response>response:get_weather{value:<|\"|>Weather is sunny<|\"|>}<tool_response|>";
        const auto object_render = gemma.apply(in);
        const auto string_render = gemma.apply(replay(args.dump()));
        expect(object_render == expected, "gemma_default_object_replay_exact");
        expect(string_render == expected, "gemma_default_string_replay_exact");
        expect(in.messages == original_messages, "gemma_apply_preserves_inputs");
        std::cout << "CAPS supports_tool_calls=" << caps.supports_tool_calls
                  << " requires_object_arguments=" << caps.requires_object_arguments
                  << " supports_tool_responses=" << caps.supports_tool_responses << '\n';
        std::cout << "DEFAULT_OBJECT " << json(object_render).dump() << '\n';
        std::cout << "DEFAULT_STRING " << json(string_render).dump() << '\n';
        if (argc == 4) {
            std::ofstream(std::string(argv[3]) + "/default-object.txt", std::ios::binary) << object_render;
            std::ofstream(std::string(argv[3]) + "/default-string.txt", std::ios::binary) << string_render;
            std::ofstream(std::string(argv[3]) + "/input-object.json", std::ios::binary) << in.messages.dump(2);
        }

        minja::chat_template_options raw;
        raw.apply_polyfills = false;
        expect(gemma.apply(in, raw) == expected, "gemma_default_matches_raw_object_render");
        bool rejected_string = false;
        try { gemma.apply(replay(args.dump()), raw); }
        catch (const std::exception & e) {
            rejected_string = contains(e.what(), "must be a JSON object (mapping), not a string");
        }
        expect(rejected_string, "gemma_raw_string_arguments_still_rejected");

        const json nested{{"city", "München 東京"},
            {"options", {{"units", "°C"}, {"days", json::array({1, 2})}, {"active", true}, {"other", nullptr}}}};
        in = replay(nested);
        const auto canonical_nested = gemma.apply(in, raw);
        expect(contains(canonical_nested, "city:<|\"|>München 東京<|\"|>"), "gemma_raw_unicode_oracle");
        expect(contains(canonical_nested, "options:{active:true,days:[1,2],other:null,units:<|\"|>°C<|\"|>}"),
               "gemma_raw_nested_oracle");
        expect(gemma.apply(in) == canonical_nested, "gemma_default_nested_object_replay");
        expect(gemma.apply(replay(nested.dump())) == canonical_nested, "gemma_default_nested_string_replay");

        in.tools = json::array({{{"type", "function"}, {"function", {
            {"name", "get_weather"}, {"description", "Weather lookup"},
            {"parameters", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}},
                            {"required", json::array({"city"})}}}}}}});
        expect(gemma.apply(in) == gemma.apply(in, raw), "gemma_default_with_tools_matches_raw");

        // Text that looks like JSON remains ordinary assistant content.
        const auto json_content = json{{"tool_calls", json::array({tool_call(args)})}}.dump();
        in = {};
        in.messages = json::array({{{"role", "user"}, {"content", "Hi"}},
                                 {{"role", "assistant"}, {"content", json_content}}});
        const auto content_render = gemma.apply(in);
        expect(contains(content_render, json_content) && !contains(content_render, "<|tool_call>"),
               "gemma_json_content_is_not_promoted");

        // A content-only template must retain the existing fallback format exactly.
        minja::chat_template fallback("{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}", "", "");
        expect(!fallback.original_caps().supports_tool_calls, "fallback_has_no_native_calls");
        const json fallback_call = {{"tool_calls", json::array({{{"name", "get_weather"}, {"arguments", args}, {"id", "call_weather_1"}}})}};
        const json fallback_response = {{"tool_response", {{"content", "Weather is sunny"}, {"tool_call_id", "call_weather_1"}}}};
        const std::string fallback_expected = "user:Weather?\nassistant:" + fallback_call.dump(2) + "\nuser:" + fallback_response.dump(2) + "\n";
        expect(fallback.apply(replay(args)) == fallback_expected, "fallback_object_calls_still_polyfilled_exactly");
        expect(fallback.apply(replay(args.dump())) == fallback_expected, "fallback_string_calls_still_polyfilled_exactly");

        minja::chat_template unquoted("{% for m in messages %}{{ m.content }}{% endfor %}argument_needle:<|\"|>print('Hello, World!')<|\"|>", "", "");
        expect(!unquoted.original_caps().supports_tool_calls, "unframed_native_looking_text_is_not_a_tool_call");

        minja::chat_template qwen(read(argv[2]), "", "");
        const auto qc = qwen.original_caps();
        expect(qc.supports_tool_calls && qc.requires_object_arguments && qc.supports_tool_responses,
               "qwen_capabilities_unchanged");
        in = replay(nested);
        const auto qwen_raw = qwen.apply(in, raw);
        expect(contains(qwen_raw, "<tool_call>") && contains(qwen_raw, "<parameter=city>"), "qwen_raw_call_oracle");
        expect(qwen.apply(in) == qwen_raw, "qwen_default_object_replay_unchanged");
        expect(qwen.apply(replay(nested.dump())) == qwen_raw, "qwen_default_string_replay_unchanged");
    } catch (const std::exception & e) {
        std::cerr << "FAIL unexpected_exception: " << e.what() << '\n';
        ++failures;
    }
    std::cout << "failures " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
