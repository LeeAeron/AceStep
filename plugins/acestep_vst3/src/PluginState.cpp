#include "PluginState.h"

namespace acestep::vst3
{
std::unique_ptr<juce::XmlElement> createStateXml(const PluginState& state)
{
    auto xml = std::make_unique<juce::XmlElement>(kStateRootTag);
    xml->setAttribute("schemaVersion", state.schemaVersion);
    xml->setAttribute("backendBaseUrl", state.backendBaseUrl);
    xml->setAttribute("prompt", state.prompt);
    xml->setAttribute("lyrics", state.lyrics);
    xml->setAttribute("durationSeconds", state.durationSeconds);
    xml->setAttribute("seed", state.seed);
    xml->setAttribute("modelPreset", toString(state.modelPreset));
    xml->setAttribute("qualityMode", toString(state.qualityMode));
    xml->setAttribute("backendStatus", toString(state.backendStatus));
    xml->setAttribute("jobStatus", toString(state.jobStatus));
    xml->setAttribute("currentTaskId", state.currentTaskId);
    xml->setAttribute("progressText", state.progressText);
    xml->setAttribute("errorMessage", state.errorMessage);
    xml->setAttribute("selectedResultSlot", state.selectedResultSlot);
    xml->setAttribute("previewFilePath", state.previewFilePath);
    xml->setAttribute("previewDisplayName", state.previewDisplayName);
    for (size_t index = 0; index < state.resultSlots.size(); ++index)
    {
        xml->setAttribute("resultSlot" + juce::String(static_cast<int>(index)),
                          state.resultSlots[index]);
        xml->setAttribute("resultFileUrl" + juce::String(static_cast<int>(index)),
                          state.resultFileUrls[index]);
        xml->setAttribute("resultLocalPath" + juce::String(static_cast<int>(index)),
                          state.resultLocalPaths[index]);
    }
    return xml;
}

std::optional<PluginState> parseStateXml(const juce::XmlElement& xml)
{
    if (!xml.hasTagName(kStateRootTag))
    {
        return std::nullopt;
    }

    PluginState state;
    state.schemaVersion = xml.getIntAttribute("schemaVersion", kCurrentStateVersion);
    state.backendBaseUrl = xml.getStringAttribute("backendBaseUrl", kDefaultBackendBaseUrl).trim();
    state.prompt = xml.getStringAttribute("prompt");
    state.lyrics = xml.getStringAttribute("lyrics");
    state.durationSeconds = xml.getIntAttribute("durationSeconds", kDefaultDurationSeconds);
    state.seed = xml.getIntAttribute("seed", kDefaultSeed);
    state.modelPreset = modelPresetFromString(xml.getStringAttribute("modelPreset"));
    state.qualityMode = qualityModeFromString(xml.getStringAttribute("qualityMode"));
    state.backendStatus = backendStatusFromString(xml.getStringAttribute("backendStatus"));
    state.jobStatus = jobStatusFromString(xml.getStringAttribute("jobStatus"));
    state.currentTaskId = xml.getStringAttribute("currentTaskId");
    state.progressText = xml.getStringAttribute("progressText");
    state.errorMessage = xml.getStringAttribute("errorMessage");
    state.selectedResultSlot =
        juce::jlimit(0, kResultSlotCount - 1, xml.getIntAttribute("selectedResultSlot", 0));
    state.previewFilePath = xml.getStringAttribute("previewFilePath");
    state.previewDisplayName = xml.getStringAttribute("previewDisplayName");
    for (size_t index = 0; index < state.resultSlots.size(); ++index)
    {
        state.resultSlots[index] =
            xml.getStringAttribute("resultSlot" + juce::String(static_cast<int>(index)));
        state.resultFileUrls[index] =
            xml.getStringAttribute("resultFileUrl" + juce::String(static_cast<int>(index)));
        state.resultLocalPaths[index] =
            xml.getStringAttribute("resultLocalPath" + juce::String(static_cast<int>(index)));
    }

    if (state.backendBaseUrl.isEmpty())
    {
        state.backendBaseUrl = kDefaultBackendBaseUrl;
    }

    return state;
}
}  // namespace acestep::vst3
