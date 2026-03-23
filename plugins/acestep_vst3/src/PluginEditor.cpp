#include "PluginEditor.h"

#include "PluginProcessor.h"

namespace acestep::vst3
{
namespace
{
constexpr int kEditorWidth = 720;
constexpr int kEditorHeight = 760;
}  // namespace

ACEStepVST3AudioProcessorEditor::ACEStepVST3AudioProcessorEditor(
    ACEStepVST3AudioProcessor& processor)
    : juce::AudioProcessorEditor(&processor), processor_(processor)
{
    setSize(kEditorWidth, kEditorHeight);
    configureLabels();
    configureEditors();
    configureSelectors();
    syncFromProcessor();
    refreshResultSelector();
    refreshStatusViews();
    startTimerHz(4);
}

ACEStepVST3AudioProcessorEditor::~ACEStepVST3AudioProcessorEditor() = default;

void ACEStepVST3AudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour::fromRGB(18, 22, 32));
    g.setColour(juce::Colour::fromRGB(44, 54, 74));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(12.0f), 12.0f, 1.5f);
}

void ACEStepVST3AudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(20);
    auto header = bounds.removeFromTop(56);
    titleLabel_.setBounds(header.removeFromTop(28));
    subtitleLabel_.setBounds(header.removeFromTop(20));

    auto top = bounds.removeFromTop(168);
    auto left = top.removeFromLeft(top.getWidth() / 2).reduced(0, 4);
    auto right = top.reduced(0, 4);

    backendUrlLabel_.setBounds(left.removeFromTop(18));
    backendUrlEditor_.setBounds(left.removeFromTop(28));
    left.removeFromTop(8);
    promptLabel_.setBounds(left.removeFromTop(18));
    promptEditor_.setBounds(left.removeFromTop(44));
    left.removeFromTop(8);
    lyricsLabel_.setBounds(left.removeFromTop(18));
    promptEditor_.setBounds(promptEditor_.getBounds());
    lyricsEditor_.setBounds(left.removeFromTop(44));

    durationLabel_.setBounds(right.removeFromTop(18));
    durationBox_.setBounds(right.removeFromTop(28));
    right.removeFromTop(8);
    seedLabel_.setBounds(right.removeFromTop(18));
    seedEditor_.setBounds(right.removeFromTop(28));
    right.removeFromTop(8);
    modelLabel_.setBounds(right.removeFromTop(18));
    modelBox_.setBounds(right.removeFromTop(28));
    right.removeFromTop(8);
    qualityLabel_.setBounds(right.removeFromTop(18));
    qualityBox_.setBounds(right.removeFromTop(28));

    auto statusArea = bounds.removeFromTop(156);
    auto statusLeft = statusArea.removeFromLeft(statusArea.getWidth() / 2).reduced(0, 4);
    auto statusRight = statusArea.reduced(0, 4);

    backendStatusTitle_.setBounds(statusLeft.removeFromTop(18));
    backendStatusBox_.setBounds(statusLeft.removeFromTop(28));
    statusLeft.removeFromTop(8);
    backendStatusValue_.setBounds(statusLeft.removeFromTop(40));
    statusLeft.removeFromTop(8);
    jobStatusTitle_.setBounds(statusLeft.removeFromTop(18));
    jobStatusValue_.setBounds(statusLeft.removeFromTop(40));

    errorTitle_.setBounds(statusRight.removeFromTop(18));
    errorValue_.setBounds(statusRight.removeFromTop(94));
    statusRight.removeFromTop(10);
    generateButton_.setBounds(statusRight.removeFromTop(30).removeFromLeft(180));

    bounds.removeFromTop(8);
    resultsLabel_.setBounds(bounds.removeFromTop(18));
    resultSlotBox_.setBounds(bounds.removeFromTop(28));

    bounds.removeFromTop(12);
    previewTitle_.setBounds(bounds.removeFromTop(18));
    previewValue_.setBounds(bounds.removeFromTop(56));
    bounds.removeFromTop(8);
    auto previewButtons = bounds.removeFromTop(30);
    choosePreviewButton_.setBounds(previewButtons.removeFromLeft(180));
    previewButtons.removeFromLeft(8);
    playPreviewButton_.setBounds(previewButtons.removeFromLeft(72));
    previewButtons.removeFromLeft(8);
    stopPreviewButton_.setBounds(previewButtons.removeFromLeft(72));
    previewButtons.removeFromLeft(8);
    clearPreviewButton_.setBounds(previewButtons.removeFromLeft(72));
    previewButtons.removeFromLeft(8);
    revealPreviewButton_.setBounds(previewButtons.removeFromLeft(120));
}

void ACEStepVST3AudioProcessorEditor::timerCallback()
{
    processor_.pumpBackendWorkflow();
    refreshResultSelector();
    refreshStatusViews();
}
}  // namespace acestep::vst3
