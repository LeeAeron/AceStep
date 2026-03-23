#include "PluginEditor.h"

#include "PluginProcessor.h"

namespace acestep::vst3
{
void ACEStepVST3AudioProcessorEditor::choosePreviewFile()
{
    previewChooser_ = std::make_unique<juce::FileChooser>("Select preview audio file",
                                                          juce::File(),
                                                          "*.wav;*.aiff;*.flac;*.ogg;*.mp3");
    previewChooser_->launchAsync(juce::FileBrowserComponent::openMode
                                     | juce::FileBrowserComponent::canSelectFiles,
                                 [this](const juce::FileChooser& chooser) {
                                     const auto file = chooser.getResult();
                                     previewChooser_.reset();
                                     if (file == juce::File())
                                     {
                                         return;
                                     }

                                     processor_.stopPreview();
                                     [[maybe_unused]] const auto loaded = processor_.loadPreviewFile(file);
                                     refreshStatusViews();
                                 });
}

void ACEStepVST3AudioProcessorEditor::playPreviewFile()
{
    processor_.playPreview();
    refreshStatusViews();
}

void ACEStepVST3AudioProcessorEditor::stopPreviewFile()
{
    processor_.stopPreview();
    refreshStatusViews();
}

void ACEStepVST3AudioProcessorEditor::clearPreviewFile()
{
    processor_.stopPreview();
    processor_.clearPreviewFile();
    refreshStatusViews();
}

void ACEStepVST3AudioProcessorEditor::revealPreviewFile()
{
    processor_.revealPreviewFile();
}
}  // namespace acestep::vst3
