#pragma once

#include <JuceHeader.h>

namespace acestep::vst3
{
enum class BackendStatus
{
    ready,
    offline,
    degraded,
};

enum class JobStatus
{
    idle,
    submitting,
    queuedOrRunning,
    succeeded,
    failed,
};

enum class ModelPreset
{
    turbo,
    standard,
    quality,
};

enum class QualityMode
{
    fast,
    balanced,
    high,
};

[[nodiscard]] juce::String toString(BackendStatus status);
[[nodiscard]] juce::String toString(JobStatus status);
[[nodiscard]] juce::String toString(ModelPreset preset);
[[nodiscard]] juce::String toString(QualityMode mode);

[[nodiscard]] BackendStatus backendStatusFromString(const juce::String& value);
[[nodiscard]] JobStatus jobStatusFromString(const juce::String& value);
[[nodiscard]] ModelPreset modelPresetFromString(const juce::String& value);
[[nodiscard]] QualityMode qualityModeFromString(const juce::String& value);
}  // namespace acestep::vst3
