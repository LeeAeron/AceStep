#include "PluginEnums.h"

namespace acestep::vst3
{
namespace
{
juce::String normalized(juce::String value)
{
    return value.trim().toLowerCase();
}
}  // namespace

juce::String toString(BackendStatus status)
{
    switch (status)
    {
        case BackendStatus::ready:
            return "Ready";
        case BackendStatus::offline:
            return "Offline";
        case BackendStatus::degraded:
            return "Degraded";
    }

    return "Ready";
}

juce::String toString(JobStatus status)
{
    switch (status)
    {
        case JobStatus::idle:
            return "Idle";
        case JobStatus::submitting:
            return "Submitting";
        case JobStatus::queuedOrRunning:
            return "Queued / Running";
        case JobStatus::succeeded:
            return "Succeeded";
        case JobStatus::failed:
            return "Failed";
    }

    return "Idle";
}

juce::String toString(ModelPreset preset)
{
    switch (preset)
    {
        case ModelPreset::turbo:
            return "Turbo";
        case ModelPreset::standard:
            return "Standard";
        case ModelPreset::quality:
            return "Quality";
    }

    return "Turbo";
}

juce::String toString(QualityMode mode)
{
    switch (mode)
    {
        case QualityMode::fast:
            return "Fast";
        case QualityMode::balanced:
            return "Balanced";
        case QualityMode::high:
            return "High";
    }

    return "Balanced";
}

BackendStatus backendStatusFromString(const juce::String& value)
{
    const auto key = normalized(value);
    if (key == "offline")
    {
        return BackendStatus::offline;
    }
    if (key == "degraded")
    {
        return BackendStatus::degraded;
    }
    return BackendStatus::ready;
}

JobStatus jobStatusFromString(const juce::String& value)
{
    const auto key = normalized(value);
    if (key == "submitting")
    {
        return JobStatus::submitting;
    }
    if (key == "queued / running" || key == "queued/running" || key == "queued_or_running")
    {
        return JobStatus::queuedOrRunning;
    }
    if (key == "succeeded")
    {
        return JobStatus::succeeded;
    }
    if (key == "failed")
    {
        return JobStatus::failed;
    }
    return JobStatus::idle;
}

ModelPreset modelPresetFromString(const juce::String& value)
{
    const auto key = normalized(value);
    if (key == "standard")
    {
        return ModelPreset::standard;
    }
    if (key == "quality")
    {
        return ModelPreset::quality;
    }
    return ModelPreset::turbo;
}

QualityMode qualityModeFromString(const juce::String& value)
{
    const auto key = normalized(value);
    if (key == "fast")
    {
        return QualityMode::fast;
    }
    if (key == "high")
    {
        return QualityMode::high;
    }
    return QualityMode::balanced;
}
}  // namespace acestep::vst3
