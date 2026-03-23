#pragma once

#include <array>
#include <memory>
#include <optional>

#include <JuceHeader.h>

#include "PluginConfig.h"
#include "PluginEnums.h"

namespace acestep::vst3
{
struct PluginState final
{
    int schemaVersion = kCurrentStateVersion;
    juce::String backendBaseUrl = kDefaultBackendBaseUrl;
    juce::String prompt;
    juce::String lyrics;
    int durationSeconds = kDefaultDurationSeconds;
    int seed = kDefaultSeed;
    ModelPreset modelPreset = ModelPreset::turbo;
    QualityMode qualityMode = QualityMode::balanced;
    BackendStatus backendStatus = BackendStatus::ready;
    JobStatus jobStatus = JobStatus::idle;
    juce::String currentTaskId;
    juce::String progressText;
    juce::String errorMessage;
    int selectedResultSlot = 0;
    std::array<juce::String, static_cast<size_t>(kResultSlotCount)> resultSlots;
    std::array<juce::String, static_cast<size_t>(kResultSlotCount)> resultFileUrls;
    std::array<juce::String, static_cast<size_t>(kResultSlotCount)> resultLocalPaths;
    juce::String previewFilePath;
    juce::String previewDisplayName;
};

[[nodiscard]] std::unique_ptr<juce::XmlElement> createStateXml(const PluginState& state);
[[nodiscard]] std::optional<PluginState> parseStateXml(const juce::XmlElement& xml);
}  // namespace acestep::vst3
