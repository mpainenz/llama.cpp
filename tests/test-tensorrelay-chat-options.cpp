#include "../tools/tensorrelay-stage-executor/chat-options.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    common_chat_templates_inputs defaults;
    tensorrelay_apply_reasoning_effort(defaults, nlohmann::ordered_json::object());
    assert(defaults.enable_thinking);
    assert(defaults.chat_template_kwargs.empty());

    for (const auto * effort : {"low", "medium", "high", "xhigh"}) {
        common_chat_templates_inputs inputs;
        tensorrelay_apply_reasoning_effort(inputs, {{"reasoning_effort", effort}});
        assert(inputs.enable_thinking);
        assert(nlohmann::ordered_json::parse(inputs.chat_template_kwargs.at("reasoning_effort")) == effort);
    }
    common_chat_templates_inputs disabled;
    tensorrelay_apply_reasoning_effort(disabled, {{"reasoning_effort", "none"}});
    assert(!disabled.enable_thinking);
    assert(disabled.chat_template_kwargs.empty());

    for (const auto & value : {nlohmann::ordered_json(nullptr), nlohmann::ordered_json(1),
                              nlohmann::ordered_json(true), nlohmann::ordered_json("unknown")}) {
        bool rejected = false;
        try {
            common_chat_templates_inputs inputs;
            tensorrelay_apply_reasoning_effort(inputs, {{"reasoning_effort", value}});
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }
    std::cout << "TensorRelay reasoning options passed\n";
}
