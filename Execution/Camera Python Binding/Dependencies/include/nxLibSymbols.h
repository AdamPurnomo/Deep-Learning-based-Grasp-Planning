#ifndef __COMMON_SYMBOLS_H__
#define __COMMON_SYMBOLS_H__

#ifdef __cli
#	define COMMON_ERROR(NAME) static String ^ err##NAME = #    NAME
#else
#	define COMMON_ERROR(NAME) static char const* const err##NAME = #    NAME
#endif /* __cli */

COMMON_ERROR(ParametersInvalid);
COMMON_ERROR(InternalError);
COMMON_ERROR(CaptureSuccessful);
COMMON_ERROR(UnknownCaptureStatus);
COMMON_ERROR(NoDestinationMemory);
COMMON_ERROR(ConversationFailed);
COMMON_ERROR(OutOfBuffers);
COMMON_ERROR(ImageLocked);
COMMON_ERROR(DeviceNotReady);
COMMON_ERROR(ImageTransferFailed);
COMMON_ERROR(DeviceTimeout);
COMMON_ERROR(DeviceCaptureFailed);
COMMON_ERROR(BufferOverrun);
COMMON_ERROR(UnknownUEyeCaptureStatus);
COMMON_ERROR(MissedImages);
COMMON_ERROR(ProjectorTimeout);
COMMON_ERROR(RingBufferIndex);
COMMON_ERROR(DiscardedImage);
COMMON_ERROR(EventBufferOverflow);
COMMON_ERROR(CaptureTimeout);
COMMON_ERROR(CaptureCancel);
COMMON_ERROR(EepromUpgradeNeeded);
COMMON_ERROR(NotSupported);
COMMON_ERROR(InvalidIpConfig);
COMMON_ERROR(ProjectorApplicationStartFailed);
COMMON_ERROR(ProjectorApplicationEndFailed);
COMMON_ERROR(ProjectorApplicationUploadFailed);
COMMON_ERROR(ProjectorNoApplication);
COMMON_ERROR(ProjectorUnknownTemperatureSensor);
COMMON_ERROR(ProjectorSignature);
COMMON_ERROR(TerminationRequested);
COMMON_ERROR(InvalidOperation);
COMMON_ERROR(InvalidGridSpacing);
COMMON_ERROR(ProfileDeserializationFailed);
COMMON_ERROR(NoOpenProfileBlock);
COMMON_ERROR(NestingLimitReached);
COMMON_ERROR(GLDriver);
COMMON_ERROR(EnsensoCalTab);
COMMON_ERROR(CUDAOutOfMemory);
COMMON_ERROR(MissingFirmware);

#undef COMMON_ERROR

#endif /*__COMMON_SYMBOLS_H__*/