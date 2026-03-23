#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace acestep::vst3
{
ACEStepVST3AudioProcessor::ACEStepVST3AudioProcessor()
    : juce::AudioProcessor(
          BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

ACEStepVST3AudioProcessor::~ACEStepVST3AudioProcessor() = default;

void ACEStepVST3AudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    preview_.prepareToPlay(sampleRate, samplesPerBlock);
}

void ACEStepVST3AudioProcessor::releaseResources()
{
    preview_.releaseResources();
}

bool ACEStepVST3AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
    {
        return false;
    }

    return layouts.getMainInputChannelSet().isDisabled();
}

void ACEStepVST3AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    buffer.clear();
    preview_.render(buffer);
}

juce::AudioProcessorEditor* ACEStepVST3AudioProcessor::createEditor()
{
    return new ACEStepVST3AudioProcessorEditor(*this);
}

bool ACEStepVST3AudioProcessor::hasEditor() const
{
    return true;
}

const juce::String ACEStepVST3AudioProcessor::getName() const
{
    return kPluginName;
}

bool ACEStepVST3AudioProcessor::acceptsMidi() const
{
    return true;
}

bool ACEStepVST3AudioProcessor::producesMidi() const
{
    return false;
}

bool ACEStepVST3AudioProcessor::isMidiEffect() const
{
    return false;
}

bool ACEStepVST3AudioProcessor::isSynth() const
{
    return true;
}

double ACEStepVST3AudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int ACEStepVST3AudioProcessor::getNumPrograms()
{
    return 1;
}

int ACEStepVST3AudioProcessor::getCurrentProgram()
{
    return 0;
}

void ACEStepVST3AudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String ACEStepVST3AudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void ACEStepVST3AudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void ACEStepVST3AudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = createStateXml(state_))
    {
        copyXmlToBinary(*xml, destData);
    }
}

void ACEStepVST3AudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml != nullptr)
    {
        if (auto parsedState = parseStateXml(*xml))
        {
            state_ = *parsedState;
            syncPreviewFromState();
        }
    }
}

const PluginState& ACEStepVST3AudioProcessor::getState() const noexcept
{
    return state_;
}

PluginState& ACEStepVST3AudioProcessor::getMutableState() noexcept
{
    return state_;
}

bool ACEStepVST3AudioProcessor::loadPreviewFile(const juce::File& file)
{
    juce::String errorMessage;
    if (!preview_.loadFile(file, errorMessage))
    {
        state_.errorMessage = errorMessage;
        return false;
    }

    state_.previewFilePath = file.getFullPathName();
    state_.previewDisplayName = file.getFileName();
    state_.errorMessage = {};
    return true;
}

void ACEStepVST3AudioProcessor::clearPreviewFile()
{
    preview_.clear();
    state_.previewFilePath = {};
    state_.previewDisplayName = {};
    state_.errorMessage = {};
}

void ACEStepVST3AudioProcessor::playPreview()
{
    if (!hasPreviewFile())
    {
        state_.errorMessage = "Load a preview file before playing it.";
        return;
    }

    state_.errorMessage = {};
    preview_.play();
}

void ACEStepVST3AudioProcessor::stopPreview()
{
    preview_.stop();
}

void ACEStepVST3AudioProcessor::revealPreviewFile() const
{
    preview_.revealToUser();
}

bool ACEStepVST3AudioProcessor::hasPreviewFile() const
{
    return preview_.hasLoadedFile();
}

bool ACEStepVST3AudioProcessor::isPreviewPlaying() const
{
    return preview_.isPlaying();
}

void ACEStepVST3AudioProcessor::requestGeneration()
{
    if (state_.prompt.trim().isEmpty())
    {
        state_.jobStatus = JobStatus::failed;
        state_.progressText = {};
        state_.errorMessage = "Prompt is required before generation.";
        return;
    }

    clearGeneratedResults();
    stopPreview();
    clearPreviewFile();
    state_.jobStatus = JobStatus::submitting;
    state_.progressText = "Submitting request...";
    state_.errorMessage = {};
}

void ACEStepVST3AudioProcessor::selectResultSlot(int index)
{
    state_.selectedResultSlot = juce::jlimit(0, kResultSlotCount - 1, index);
    const auto& localPath = state_.resultLocalPaths[static_cast<size_t>(state_.selectedResultSlot)];
    if (localPath.isNotEmpty())
    {
        [[maybe_unused]] const auto loaded = loadPreviewFile(juce::File(localPath));
    }
}

void ACEStepVST3AudioProcessor::pumpBackendWorkflow()
{
    std::optional<BackendTaskResult> completedTask;
    {
        const juce::ScopedLock lock(backendTaskLock_);
        if (completedBackendTask_.has_value())
        {
            completedTask = std::move(completedBackendTask_);
            completedBackendTask_.reset();
        }
    }

    if (completedTask.has_value())
    {
        applyCompletedTask(*completedTask);
    }

    if (backendTaskRunning_.load())
    {
        return;
    }

    if (pendingPreviewDownloadSlot_.has_value())
    {
        schedulePreviewDownload(*pendingPreviewDownloadSlot_);
        pendingPreviewDownloadSlot_.reset();
        return;
    }

    const auto now = juce::Time::getMillisecondCounter();
    if (state_.jobStatus == JobStatus::submitting)
    {
        scheduleGenerationStart();
        return;
    }

    if (state_.jobStatus == JobStatus::queuedOrRunning && state_.currentTaskId.isNotEmpty()
        && now - lastPollRequestAtMs_ >= 1500)
    {
        scheduleGenerationPoll();
        return;
    }

    if (state_.jobStatus == JobStatus::idle || state_.jobStatus == JobStatus::failed
        || state_.jobStatus == JobStatus::succeeded)
    {
        if (lastHealthCheckedBaseUrl_ != state_.backendBaseUrl
            || now - lastHealthCheckAtMs_ >= 5000)
        {
            scheduleHealthCheck();
        }
    }
}

void ACEStepVST3AudioProcessor::scheduleHealthCheck()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    lastHealthCheckedBaseUrl_ = state_.backendBaseUrl;
    lastHealthCheckAtMs_ = juce::Time::getMillisecondCounter();
    const auto baseUrl = state_.backendBaseUrl;
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, baseUrl]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::healthCheck;
        taskResult.health = backendClient_.checkHealth(baseUrl);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::scheduleGenerationStart()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    const auto stateSnapshot = state_;
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, stateSnapshot]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::submitGeneration;
        taskResult.generationStart = backendClient_.startGeneration(stateSnapshot);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::scheduleGenerationPoll()
{
    if (backendTaskRunning_.exchange(true))
    {
        return;
    }

    const auto baseUrl = state_.backendBaseUrl;
    const auto taskId = state_.currentTaskId;
    lastPollRequestAtMs_ = juce::Time::getMillisecondCounter();
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>([this, baseUrl, taskId]() {
        BackendTaskResult taskResult;
        taskResult.kind = BackendTaskKind::pollGeneration;
        taskResult.generationPoll = backendClient_.pollGeneration(baseUrl, taskId);
        const juce::ScopedLock lock(backendTaskLock_);
        completedBackendTask_ = std::move(taskResult);
        backendTaskRunning_.store(false);
        return juce::ThreadPoolJob::jobHasFinished;
    }));
}

void ACEStepVST3AudioProcessor::schedulePreviewDownload(int slotIndex)
{
    if (backendTaskRunning_.exchange(true))
    {
        pendingPreviewDownloadSlot_ = slotIndex;
        return;
    }

    const auto baseUrl = state_.backendBaseUrl;
    const auto remoteFileUrl = state_.resultFileUrls[static_cast<size_t>(slotIndex)];
    backendThreadPool_.addJob(std::function<juce::ThreadPoolJob::JobStatus()>(
        [this, baseUrl, remoteFileUrl, slotIndex]() {
            BackendTaskResult taskResult;
            taskResult.kind = BackendTaskKind::downloadPreview;
            taskResult.previewDownload =
                backendClient_.downloadPreviewFile(baseUrl, remoteFileUrl, slotIndex);
            const juce::ScopedLock lock(backendTaskLock_);
            completedBackendTask_ = std::move(taskResult);
            backendTaskRunning_.store(false);
            return juce::ThreadPoolJob::jobHasFinished;
        }));
}

void ACEStepVST3AudioProcessor::applyCompletedTask(const BackendTaskResult& taskResult)
{
    switch (taskResult.kind)
    {
        case BackendTaskKind::healthCheck:
            state_.backendStatus = taskResult.health.status;
            if (taskResult.health.status == BackendStatus::ready
                && state_.jobStatus != JobStatus::failed)
            {
                state_.errorMessage = {};
            }
            else if (taskResult.health.status != BackendStatus::ready
                     && state_.jobStatus == JobStatus::idle)
            {
                state_.errorMessage = taskResult.health.errorMessage;
            }
            return;
        case BackendTaskKind::submitGeneration:
            if (!taskResult.generationStart.succeeded)
            {
                state_.jobStatus = JobStatus::failed;
                state_.progressText = {};
                state_.errorMessage = taskResult.generationStart.errorMessage;
                state_.currentTaskId = {};
                return;
            }

            state_.backendStatus = BackendStatus::ready;
            state_.jobStatus = JobStatus::queuedOrRunning;
            state_.currentTaskId = taskResult.generationStart.taskId;
            state_.progressText = "Task started: " + state_.currentTaskId;
            state_.errorMessage = {};
            return;
        case BackendTaskKind::pollGeneration:
            state_.jobStatus = taskResult.generationPoll.status;
            state_.progressText = taskResult.generationPoll.progressText;
            if (taskResult.generationPoll.status == JobStatus::failed)
            {
                state_.errorMessage = taskResult.generationPoll.errorMessage;
                state_.currentTaskId = {};
                return;
            }

            if (taskResult.generationPoll.status != JobStatus::succeeded)
            {
                return;
            }

            state_.currentTaskId = {};
            state_.errorMessage = {};
            for (int index = 0; index < kResultSlotCount; ++index)
            {
                const auto& slot = taskResult.generationPoll.resultSlots[static_cast<size_t>(index)];
                state_.resultSlots[static_cast<size_t>(index)] = slot.label;
                state_.resultFileUrls[static_cast<size_t>(index)] = slot.remoteFileUrl;
            }
            state_.selectedResultSlot = 0;
            if (state_.resultFileUrls[0].isNotEmpty())
            {
                pendingPreviewDownloadSlot_ = 0;
            }
            else
            {
                state_.errorMessage = "Task finished but no audio file was returned.";
            }
            return;
        case BackendTaskKind::downloadPreview:
            if (!taskResult.previewDownload.succeeded)
            {
                state_.errorMessage = taskResult.previewDownload.errorMessage;
                return;
            }

            if (taskResult.previewDownload.slotIndex >= 0
                && taskResult.previewDownload.slotIndex < kResultSlotCount)
            {
                const auto slotIndex = static_cast<size_t>(taskResult.previewDownload.slotIndex);
                state_.resultLocalPaths[slotIndex] = taskResult.previewDownload.localFilePath;
                if (state_.selectedResultSlot == taskResult.previewDownload.slotIndex)
                {
                    [[maybe_unused]] const auto loaded =
                        loadPreviewFile(juce::File(taskResult.previewDownload.localFilePath));
                }
            }
            return;
        case BackendTaskKind::none:
            return;
    }
}

void ACEStepVST3AudioProcessor::clearGeneratedResults()
{
    state_.currentTaskId = {};
    state_.progressText = {};
    state_.selectedResultSlot = 0;
    pendingPreviewDownloadSlot_.reset();
    for (int index = 0; index < kResultSlotCount; ++index)
    {
        state_.resultSlots[static_cast<size_t>(index)] = {};
        state_.resultFileUrls[static_cast<size_t>(index)] = {};
        state_.resultLocalPaths[static_cast<size_t>(index)] = {};
    }
}

void ACEStepVST3AudioProcessor::syncPreviewFromState()
{
    preview_.clear();
    if (state_.previewFilePath.isEmpty())
    {
        return;
    }

    juce::String errorMessage;
    const juce::File previewFile(state_.previewFilePath);
    if (!preview_.loadFile(previewFile, errorMessage))
    {
        state_.errorMessage = errorMessage;
        return;
    }

    state_.previewDisplayName = previewFile.getFileName();
}
}  // namespace acestep::vst3

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new acestep::vst3::ACEStepVST3AudioProcessor();
}
