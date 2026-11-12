#pragma once

#include "chat.h"
#include "nlohmann/json.hpp"
#include <stdexcept>

inline void tensorrelay_apply_reasoning_effort(common_chat_templates_inputs & inputs, const nlohmann::ordered_json & body) {
    inputs.enable_thinking = true;
    if (!body.contains("reasoning_effort")) {
        return;
    }
    const auto & value = body.at("reasoning_effort");
    if (!value.is_string()) {
        throw std::invalid_argument("reasoning_effort must be a string");
    }
    const auto effort = value.get<std::string>();
    if (effort == "none") {
        inputs.enable_thinking = false;
    } else if (effort == "low" || effort == "medium" || effort == "high" || effort == "xhigh") {
        inputs.chat_template_kwargs["reasoning_effort"] = value.dump();
    } else {
        throw std::invalid_argument("reasoning_effort must be none, low, medium, high or xhigh");
    }
}
