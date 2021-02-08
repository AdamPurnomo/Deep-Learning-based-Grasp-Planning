#ifndef __NXLIB_CONSTANTS_H__
#define __NXLIB_CONSTANTS_H__

#ifndef __cli
#	if defined(_WIN32) || defined(__WINDOWS__)
#		define NXLIBERR __int32
#		define NXLIBBOOL __int32
#		define NXLIBINT __int32
#		define NXLIBDOUBLE double
#	else /* defined (_WIN32) */
#		include <stdint.h>
#		define NXLIBERR int32_t
#		define NXLIBBOOL int32_t
#		define NXLIBINT int32_t
#		define NXLIBDOUBLE double
#	endif

#	define NXLIBSTR const char*
#	define NXLIBFALSE 0
#	define NXLIBTRUE 1

#	define NxLibItemSeparator '/'
#	define NxLibIndexEscapeChar '\\'
#	define NxLibItemForbiddenChars "\r\n\"/\\\0"
#endif /* __cli */

#ifdef __cli
#	undef __COMMON_NODENAMES_H__

#	undef NxLibItemSeparator
#	undef NxLibIndexEscapeChar
#	undef NxLibItemForbiddenChars
#	define NxLibItemSeparator "/"
#	define NxLibIndexEscapeChar "\\"
#	define NxLibItemForbiddenChars "\r\n\"/\\\0"

#	undef NXLIB_ITEM
#	undef NXLIB_COMMAND
#	undef NXLIB_VALUE
#	undef NXLIB_ERROR
#	undef NXLIB_INT_CONST
#	undef NXLIB_VALUE_
#	undef NXLIB_ERROR_
#	define NXLIB_ITEM(NAME) static String ^ itm##NAME = #    NAME
#	define NXLIB_COMMAND(NAME) static String ^ cmd##NAME = #    NAME
#	define NXLIB_VALUE(NAME) static String ^ val##NAME = #    NAME
#	define NXLIB_ERROR(NAME) static String ^ err##NAME = #    NAME
#	define NXLIB_VALUE_(NAME, VALUE) static String ^ val##NAME = VALUE
#	define NXLIB_ERROR_(NAME, VALUE) static String ^ err##NAME = VALUE
#	define NXLIB_INT_CONST(NAME, VALUE) static int NAME = VALUE
#	ifdef NXLIB_APIERROR
#		undef NXLIB_APIERROR
#	endif
#	define NXLIB_APIERROR(NAME, VALUE) static int const NAME = VALUE

static short const InvalidDisparityScaled = (short)0x8000;

enum class DebugLevel { nxdInherit = 0, nxdInfo = 1, nxdDebug = 2, nxdTrace = 3 };
#else
#	ifndef OVERRIDE_NXLIB_CONSTANTS
#		define NXLIB_ITEM(NAME) static char const* const itm##NAME = #        NAME
#		define NXLIB_COMMAND(NAME) static char const* const cmd##NAME = #        NAME
#		define NXLIB_VALUE(NAME) static char const* const val##NAME = #        NAME
#		define NXLIB_ERROR(NAME) static char const* const err##NAME = #        NAME
#		define NXLIB_VALUE_(NAME, VALUE) static char const* const val##NAME = VALUE
#		define NXLIB_ERROR_(NAME, VALUE) static char const* const err##NAME = VALUE
#		define NXLIB_INT_CONST(NAME, VALUE) static int const NxLib##NAME = VALUE
#		ifdef NXLIB_APIERROR
#			undef NXLIB_APIERROR
#		endif
#		define NXLIB_APIERROR(NAME, VALUE) static int const NxLib##NAME = VALUE
#	endif /* OVERRIDE_NXLIB_CONSTANTS */

static short const NxLibInvalidDisparityScaled = (short)0x8000;

enum NxLibDebugLevel { nxdInherit = 0, nxdInfo = 1, nxdDebug = 2, nxdTrace = 3 };
#endif     /* __cli */

#ifdef ENSENSO_DEV
#	include "dataStructures/dataStructures/commonNodeNames.h"
#else
#	include "nxLibCommonNodeNames.h"
#endif

NXLIB_ITEM(Debug);
NXLIB_ITEM(Level);
NXLIB_ITEM(BufferSize);
NXLIB_ITEM(BufferCount);
NXLIB_ITEM(StaticBufferCount);
NXLIB_ITEM(InfoTimeout);
NXLIB_ITEM(Destination); // Deprecated
NXLIB_ITEM(FileOutput);
// NXLIB_ITEM(Enabled);
NXLIB_ITEM(WriteFrequency);
NXLIB_ITEM(FilePrefix);
// NXLIB_ITEM(FolderPath);
NXLIB_ITEM(MaxTotalSize);
NXLIB_ITEM(MaxFileSize);

NXLIB_ITEM(Version);
NXLIB_ITEM(NxLib);
NXLIB_ITEM(Year);
NXLIB_ITEM(Month);
NXLIB_ITEM(Day);
NXLIB_ITEM(Major);
NXLIB_ITEM(Minor);
NXLIB_ITEM(Build);
NXLIB_ITEM(Hash);
NXLIB_ITEM(UEye);
NXLIB_ITEM(CodeMeter);
NXLIB_ITEM(HasLicense);
NXLIB_ITEM(SystemInfo);
NXLIB_ITEM(Architecture);
NXLIB_ITEM(OperatingSystem);

// NXLIB_ITEM(Cameras);
NXLIB_ITEM(ByEepromId);
NXLIB_ITEM(BySerialNo);
NXLIB_ITEM(SerialNumber);
NXLIB_ITEM(ModelName);
NXLIB_ITEM(EepromId);
NXLIB_ITEM(Status);
// NXLIB_ITEM(Info);
NXLIB_ITEM(Open);
NXLIB_ITEM(Available);
NXLIB_ITEM(ValidIpAddress);
NXLIB_ITEM(ValidFirmware);
NXLIB_ITEM(ValidCameraFirmware);
NXLIB_ITEM(ValidProjectorFirmware);
NXLIB_ITEM(Paired);
NXLIB_ITEM(Calibrated);
NXLIB_ITEM(Overtemperature);
NXLIB_ITEM(HardwareFailure);
NXLIB_ITEM(DynamicCalibrationLimitReached);
NXLIB_ITEM(MinimumVoltage);
NXLIB_ITEM(LocalEepromFile);
NXLIB_ITEM(LowBandwidth);
NXLIB_ITEM(ImageFolder);
NXLIB_ITEM(Temperature);
NXLIB_ITEM(Board);
NXLIB_ITEM(LED);
NXLIB_ITEM(CPU);
// NXLIB_ITEM(Type);
NXLIB_ITEM(Sensor);
// NXLIB_ITEM(PixelSize);
NXLIB_ITEM(GlobalShutter);
// NXLIB_ITEM(Size);
NXLIB_ITEM(MaximumTransmissionUnit);
NXLIB_ITEM(Port);
NXLIB_ITEM(Bandwidth);
NXLIB_ITEM(IpAddress);
NXLIB_ITEM(IpSubnetMask);
NXLIB_ITEM(Gateway);
NXLIB_ITEM(DefaultGateway); // Deprecated: use Gateway instead
NXLIB_ITEM(NetworkAdapter);
NXLIB_ITEM(NetworkAdapterIpAddress);    // Deprecated, now a subnode of NetworkAdapter.
NXLIB_ITEM(NetworkAdapterIpSubnetMask); // Deprecated, now a subnode of NetworkAdapter.
NXLIB_ITEM(MAC);
NXLIB_ITEM(ReversePathFiltering);
NXLIB_ITEM(DHCP);
// NXLIB_ITEM(Type);
NXLIB_ITEM(FirmwareVersion);
NXLIB_ITEM(Bootloader);
NXLIB_ITEM(Application);
NXLIB_ITEM(Updater);
NXLIB_ITEM(Remote);

NXLIB_ITEM(PatternBuffer);

NXLIB_COMMAND(Break);
NXLIB_COMMAND(Open);
NXLIB_ITEM(LoadCalibration);
NXLIB_ITEM(GenerateEmptyCalibration);
NXLIB_ITEM(AllowFirmwareUpload); // Deprecated: use parameter FirmwareUpload/Camera instead
NXLIB_ITEM(FirmwareUpload);
// NXLIB_ITEM(Projector)
// NXLIB_ITEM(Camera)
NXLIB_ITEM(KeepAliveTimeout);
NXLIB_COMMAND(Close);
NXLIB_ITEM(Unsafe);
NXLIB_COMMAND(ClearImages);
NXLIB_COMMAND(Capture);
NXLIB_ITEM(InitialTrigger);
NXLIB_ITEM(FinalTrigger);
NXLIB_ITEM(WaitFor);
NXLIB_ITEM(Timeout);
NXLIB_ITEM(Operation);
NXLIB_ITEM(WaitForProjector);
NXLIB_ITEM(ClearImageBufferOnTrigger);
NXLIB_ITEM(Retrieved);
NXLIB_ITEM(PacketsResent);
NXLIB_ITEM(CRCErrorCount);
NXLIB_ITEM(CaptureEvents);
// NXLIB_ITEM(Left);
// NXLIB_ITEM(Right);
NXLIB_ITEM(Symbol);
NXLIB_ITEM(Message);
NXLIB_ITEM(Timestamp);

NXLIB_COMMAND(Retrieve);
NXLIB_COMMAND(Trigger);
NXLIB_ITEM(Subset);
NXLIB_COMMAND(RectifyImages);
NXLIB_COMMAND(ComputeDisparityMap);
NXLIB_ITEM(MarkFilterRegions); // Unused: feature has been removed
NXLIB_COMMAND(ComputePointMap);
NXLIB_COMMAND(ComputeNormals);
NXLIB_COMMAND(ConvertTransformation);
NXLIB_ITEM(SplitRotation);
NXLIB_VALUE(XYZ);
NXLIB_VALUE(ZYX);
NXLIB_COMMAND(ChainTransformations);
NXLIB_ITEM(Transformations);
NXLIB_COMMAND(ComputeImageContrast);
NXLIB_ITEM(Region);
NXLIB_ITEM(UseBufferedPatterns);
NXLIB_ITEM(BlurMinMax);
NXLIB_COMMAND(SaveModel);
NXLIB_ITEM(Texture);
NXLIB_COMMAND(SaveImage);
NXLIB_ITEM(Node);
// NXLIB_ITEM(Filename);
NXLIB_COMMAND(LoadImage);
NXLIB_ITEM(ForceGrayscale);
NXLIB_COMMAND(SaveText);
NXLIB_ITEM(Content);
NXLIB_COMMAND(LoadText);
NXLIB_COMMAND(DiscardPatterns);
NXLIB_COMMAND(ReducePatterns); // Deprecated
NXLIB_ITEM(DrawOnly);
NXLIB_ITEM(ShowPattern);
NXLIB_COMMAND(LoadCalibration);
NXLIB_ITEM(EepromFormat);
NXLIB_ITEM(FromDynamic);
NXLIB_COMMAND(StoreCalibration);
NXLIB_ITEM(Force);
NXLIB_ITEM(OverwriteWithDynamic); // Deprecated
// NXLIB_ITEM(Calibration);
// NXLIB_ITEM(Link);
NXLIB_ITEM(DefaultParameters);
NXLIB_ITEM(DynamicOffsets);
NXLIB_ITEM(MaxEepromFormat);
NXLIB_COMMAND(LoadUEyeParameterSet);
NXLIB_COMMAND(Recalibrate);
NXLIB_COMMAND(Calibrate);
NXLIB_VALUE(Pattern);
NXLIB_COMMAND(CalibrateInBackground);
NXLIB_COMMAND(CalibrateUpdatePatterns);
NXLIB_COMMAND(CalibratePattern);
NXLIB_COMMAND(GetPatternBuffers); // Deprecated
NXLIB_COMMAND(SetPatternBuffers); // Deprecated
NXLIB_COMMAND(GetConstants);
NXLIB_COMMAND(PatternBufferInfo);
NXLIB_ITEM(ConnectedCameras);
NXLIB_ITEM(ConnectedPatterns);
NXLIB_COMMAND(GetPatternBuffer);
NXLIB_COMMAND(SetPatternBuffer);
NXLIB_COMMAND(FilterPatternBuffer);
NXLIB_COMMAND(ReducePatternBuffer);
// NXLIB_ITEM(Mode)
NXLIB_VALUE(ImagePosition);
NXLIB_ITEM(Reduced);
NXLIB_COMMAND(ProjectPattern);
NXLIB_COMMAND(CollectPlanePoints);
NXLIB_ITEM(PersistentOverlay);
NXLIB_COMMAND(CollectPattern);
NXLIB_ITEM(DecodeData);
NXLIB_ITEM(IgnoreEnsensoPatternEncoding);
NXLIB_ITEM(UpdateGlobalPatternData);
NXLIB_ITEM(GlobalPatternDataUpdated);
NXLIB_ITEM(MarkOnly); // Deprecated
NXLIB_ITEM(MeasureContrast);
NXLIB_ITEM(ReturnAllPattern);
NXLIB_ITEM(DrawAxes);
NXLIB_ITEM(DrawOrigin);
NXLIB_ITEM(Buffer);
NXLIB_ITEM(Brightness);
NXLIB_ITEM(Refinement);
// NXLIB_VALUE(None);
NXLIB_VALUE(Immediate);
NXLIB_VALUE(Delayed); // Deprecated
NXLIB_ITEM(Downsample);
NXLIB_COMMAND(CalibrateWorkspace);
NXLIB_ITEM(AlignAxis);
NXLIB_ITEM(Plane);
NXLIB_COMMAND(CalibrateHandEye);
NXLIB_ITEM(Angles);
NXLIB_ITEM(Poses);
NXLIB_ITEM(Iterations);
NXLIB_ITEM(Tolerance);
NXLIB_ITEM(UsePosesOnly);
NXLIB_ITEM(UseTriangulatedPoints);
NXLIB_ITEM(Setup);
NXLIB_VALUE(Moving);
NXLIB_VALUE(Fixed);
NXLIB_ITEM(MeasureCalibration);
NXLIB_ITEM(PoseError);
NXLIB_COMMAND(EstimateDisparitySettings);
NXLIB_COMMAND(EstimatePatternPose);
NXLIB_ITEM(Relative);
NXLIB_ITEM(ReprojectionErrorScale);
NXLIB_ITEM(Average);
NXLIB_ITEM(Index);
NXLIB_ITEM(Recalibrate);
NXLIB_ITEM(EstimateGridSpacing);
NXLIB_ITEM(Flags);
NXLIB_ITEM(StereoIntrinsic);
NXLIB_ITEM(StereoExtrinsic);
NXLIB_ITEM(MonoIntrinsic);
NXLIB_COMMAND(MeasureCalibration);
NXLIB_ITEM(Direction);
NXLIB_VALUE(RightToLeft);
NXLIB_VALUE(LeftToRight);
NXLIB_COMMAND(RenderView);
NXLIB_ITEM(ColorRepetitionDistance);
NXLIB_ITEM(ColorOffset);
NXLIB_ITEM(UseStereoTextures);
// NXLIB_ITEM(ViewPose);
// NXLIB_ITEM(ShowSurface);
// NXLIB_ITEM(ShowCameras);
// NXLIB_ITEM(ShowGrid);
// NXLIB_ITEM(ShowUserDefinedModels);
// NXLIB_ITEM(Size);
// NXLIB_ITEM(SurfaceConnectivity);

// NXLIB_ITEM(Normal);
NXLIB_ITEM(Subsampling);
NXLIB_COMMAND(RenderDepthMap); // Deprecated since 1.2; renamed to RenderPointMap
NXLIB_COMMAND(RenderPointMap);
NXLIB_ITEM(RenderGroundTruth);
// NXLIB_ITEM(PixelSize);
// NXLIB_ITEM(Size);
// NXLIB_ITEM(Scaling);
// NXLIB_ITEM(UseOpenGL);
// NXLIB_ITEM(ViewPose);
// NXLIB_ITEM(SurfaceConnectivity);
NXLIB_COMMAND(ClearImageBuffer);
NXLIB_COMMAND(ClearOverlay);
NXLIB_ITEM(ShowRectifiedArea);
NXLIB_COMMAND(VisualizePatternBuffer);
NXLIB_ITEM(ProgressMode);
NXLIB_ITEM(ProgressFactor);
NXLIB_ITEM(ProgressMask);
NXLIB_ITEM(ProgressMaskFactor);
NXLIB_ITEM(ProgressGrid);
NXLIB_COMMAND(GetRawCalibrationData);
NXLIB_COMMAND(GetModelInfo);
NXLIB_ITEM(WorldCoordinates);
NXLIB_ITEM(Disparity);
NXLIB_ITEM(Distance);
NXLIB_ITEM(ListSensors);
NXLIB_ITEM(ListLenses);
NXLIB_ITEM(ListModels);
NXLIB_ITEM(ListPatterns);
NXLIB_ITEM(ListCalTabs);
NXLIB_ITEM(Blur);
NXLIB_ITEM(Focus);
NXLIB_ITEM(FocalLength);
NXLIB_ITEM(Aperture);
NXLIB_ITEM(PixelPitch);
NXLIB_ITEM(DisparityAccuracy);
// NXLIB_ITEM(Pattern);
NXLIB_ITEM(OuterSize);
// NXLIB_ITEM(GridSpacing);
// NXLIB_ITEM(GridSize);
// NXLIB_ITEM(Thickness);
NXLIB_ITEM(Sensors);
NXLIB_ITEM(Lenses);
NXLIB_ITEM(Models);
NXLIB_ITEM(CalTabs);
NXLIB_COMMAND(FitPrimitive);
// NXLIB_ITEM(Points);
NXLIB_ITEM(Primitive);
NXLIB_ITEM(InlierThreshold);
// NXLIB_ITEM(Iterations);
NXLIB_ITEM(BoundingBox);
NXLIB_ITEM(InlierFraction);
NXLIB_ITEM(InlierCount);
NXLIB_ITEM(Score);
NXLIB_ITEM(FailureProbability);
NXLIB_ITEM(Center);
NXLIB_ITEM(Normal);
NXLIB_ITEM(Residual);
NXLIB_ITEM(Count);
// NXLIB_ITEM(Radius);
//		NXLIB_ITEM(Min);
//		NXLIB_ITEM(Max);
NXLIB_COMMAND(AdapterConfiguration);
NXLIB_ITEM(CalculateIp);
NXLIB_ITEM(Temporary);
NXLIB_ITEM(NetworkType);
NXLIB_VALUE(NetworkTypeA);
NXLIB_VALUE(NetworkTypeB);
NXLIB_VALUE(NetworkTypeC);
NXLIB_COMMAND(EthernetConfiguration);
NXLIB_ITEM(Configuration);
NXLIB_ITEM(Method);
NXLIB_ITEM(Blind);
NXLIB_ITEM(MacAddresses);
NXLIB_COMMAND(AdjustExposureAndGain);
NXLIB_ITEM(AdjustExposure);
NXLIB_ITEM(AdjustGain);
NXLIB_ITEM(Apply);
NXLIB_COMMAND(GenerateCalibrationPattern);
// NXLIB_ITEM(Filename);
NXLIB_ITEM(Encoding);
NXLIB_ITEM(ImageSize);
NXLIB_ITEM(Patterns);
// NXLIB_ITEM(GridSpacing);
// NXLIB_ITEM(Thickness);
NXLIB_ITEM(Text);
NXLIB_COMMAND(CreateCamera);
// NXLIB_ITEM(SerialNumber);
// NXLIB_ITEM(Type);
// NXLIB_ITEM(Left);
// NXLIB_ITEM(SerialNumber);
NXLIB_ITEM(Rotate);
// NXLIB_ITEM(Right);
NXLIB_ITEM(FolderPath);
NXLIB_ITEM(CalibrationFile);
NXLIB_ITEM(PatternType);
// NXLIB_ITEM(Sensor);
NXLIB_ITEM(Lens);
// NXLIB_ITEM(Baseline);
// NXLIB_ITEM(Vergence);
NXLIB_ITEM(ProjectorPower);
NXLIB_ITEM(Master);
NXLIB_ITEM(AsynchronouslyTriggered);
NXLIB_ITEM(WiringTest);
NXLIB_ITEM(AutoSwap);
NXLIB_COMMAND(Synchronize);
// NXLIB_ITEM(Cameras);
NXLIB_ITEM(Nodes);
NXLIB_COMMAND(DeleteCamera);
NXLIB_COMMAND(InternalPwm);
NXLIB_COMMAND(SimulatePhysics);
// NXLIB_ITEM(Time);
NXLIB_ITEM(ResetClock);

NXLIB_ITEM(Calibration);
NXLIB_ITEM(Monocular);
NXLIB_ITEM(Camera);
NXLIB_ITEM(Distortion);
// NXLIB_ITEM(BinningShift);
NXLIB_ITEM(Stereo);
NXLIB_ITEM(Rectification); // Hidden node to disable rectification; this allows to load and use rectified images in the
                           // Raw image nodes. Introduced for compatibility with versions <1.2 which allowed saving of
                           // rectified images.
NXLIB_ITEM(Baseline);
NXLIB_ITEM(DisparityMapOffset);
NXLIB_ITEM(Reprojection);
// NXLIB_ITEM(Rotation);
// NXLIB_ITEM(Angle);
NXLIB_ITEM(Vergence);
NXLIB_ITEM(HalfVergence);
// NXLIB_ITEM(Epipolar);
NXLIB_ITEM(OpticalAxis);
NXLIB_ITEM(Skew);
NXLIB_ITEM(Dynamic);
// NXLIB_ITEM(Monocular);
// NXLIB_ITEM(Stereo);
// NXLIB_ITEM(Vergence);
// NXLIB_ITEM(Epipolar);
NXLIB_ITEM(ForcedRawImageSize);
NXLIB_ITEM(ForcedRectifiedImageSize);
NXLIB_ITEM(RawAoiIncrements);
NXLIB_ITEM(RectifiedAoiIncrements);
NXLIB_ITEM(Top);
NXLIB_ITEM(CalibrationHistory);
// NXLIB_ITEM(Epipolar);
// NXLIB_ITEM(Time);
NXLIB_COMMAND(AddPatternBufferView);
//	NXLIB_ITEM(Name);
NXLIB_COMMAND(RemovePatternBufferView);
// NXLIB_ITEM(Name);
NXLIB_COMMAND(Flash);
// NXLIB_ITEM(Projector);
NXLIB_ITEM(Glow);
NXLIB_COMMAND(SetStatusLeds);
// NXLIB_ITEM(Left);
// NXLIB_ITEM(Right);
NXLIB_ITEM(Ext);
NXLIB_ITEM(Green);
NXLIB_ITEM(Yellow);
NXLIB_ITEM(Automatic);
NXLIB_COMMAND(NetworkControl);
NXLIB_ITEM(StartDaemon);
NXLIB_ITEM(StopDaemon);
NXLIB_ITEM(RestartDaemon);
NXLIB_ITEM(Running);
NXLIB_ITEM(USB);
NXLIB_ITEM(CheckDaemon);
NXLIB_ITEM(EnableIpFilter);
NXLIB_ITEM(RefreshUeyeAdapters);
NXLIB_COMMAND(UpdateFirmware);
NXLIB_COMMAND(Loop);
NXLIB_ITEM(IgnoredErrorSymbols);

NXLIB_ITEM(ViewPose);
NXLIB_ITEM(PixelSize);
NXLIB_ITEM(ZBufferOnly);
NXLIB_ITEM(FillXYCoordinates);
NXLIB_ITEM(PatternPose);
NXLIB_ITEM(DefinedPose);

// NXLIB_ITEM(Execute);
//	NXLIB_ITEM(Default);
// NXLIB_ITEM(Command);
// NXLIB_ITEM(Parameters);
// NXLIB_ITEM(Result);
// NXLIB_ITEM(Time);
NXLIB_ITEM(TimePrepare);
NXLIB_ITEM(TimeExecute);
NXLIB_ITEM(TimeFinalize);
NXLIB_ITEM(ErrorSymbol);
NXLIB_ITEM(ErrorText);
NXLIB_ITEM(Progress);
NXLIB_ITEM(MonocularCalibration);
NXLIB_ITEM(StereoCalibration);
NXLIB_ITEM(PersistentParameters);
// NXLIB_ITEM(Status);
NXLIB_ITEM(Messages);
NXLIB_ITEM(LatestMessage);

NXLIB_ITEM(PixelClock);
NXLIB_ITEM(Projector);
NXLIB_ITEM(FrontLight);
NXLIB_ITEM(Hdr);
NXLIB_ITEM(HardwareGamma);
NXLIB_ITEM(GainBoost);
NXLIB_ITEM(Exposure);
NXLIB_ITEM(Gain);
NXLIB_ITEM(MaxFlashTime);
NXLIB_ITEM(MaxGain);
// NXLIB_ITEM(Focus);
NXLIB_ITEM(AutoExposure);
NXLIB_ITEM(AutoGain);
NXLIB_ITEM(AutoFocus);
NXLIB_ITEM(TriggerDelay);
NXLIB_ITEM(TriggerMode);
NXLIB_ITEM(FlexView);
NXLIB_ITEM(IO);
NXLIB_ITEM(Output);
NXLIB_ITEM(Mode);
NXLIB_ITEM(Frequency);
NXLIB_ITEM(DutyCycle);
NXLIB_ITEM(Duration);
NXLIB_ITEM(Driver);
NXLIB_ITEM(Sink);
NXLIB_ITEM(Source);
NXLIB_ITEM(Input);
NXLIB_ITEM(TargetBrightness);
NXLIB_ITEM(BlackLevelOffsetCalibration);
NXLIB_ITEM(BlackLevelOffset);
NXLIB_ITEM(AbsoluteBlackLevelOffset);
NXLIB_ITEM(AutoBlackLevel);
NXLIB_ITEM(UseDisparityMapAreaOfInterest);
NXLIB_ITEM(FlashDelay);
NXLIB_ITEM(Triggered);
NXLIB_ITEM(MultiExposureFactor);
NXLIB_ITEM(ImageSet);
NXLIB_ITEM(NumberOfImageSets);
NXLIB_ITEM(ImageName);
NXLIB_ITEM(UseRecalibrator);
NXLIB_ITEM(WaitForRecalibration);
NXLIB_ITEM(FollowDynamicOffsets);
NXLIB_ITEM(ImageBuffer);
NXLIB_ITEM(OverflowPolicy);
// NXLIB_ITEM(Aperture);
NXLIB_ITEM(FocusDistance);
NXLIB_ITEM(NoiseLevel);
NXLIB_ITEM(Vignetting);
// NXLIB_ITEM(ProjectorPower);
NXLIB_ITEM(HighQualityRendering);
NXLIB_ITEM(ProjectorPattern);
NXLIB_ITEM(TransportLayer);
NXLIB_ITEM(BandwidthLimit);
NXLIB_ITEM(InternalTrigger);
NXLIB_ITEM(DownloadImages);
NXLIB_ITEM(ImageDownloadLimit);
NXLIB_ITEM(ProjectorMinimumDutyCycle);

NXLIB_ITEM(Size);
// NXLIB_ITEM(Scaling);
// NXLIB_ITEM(Binning);

NXLIB_ITEM(Interface);
NXLIB_ITEM(Ethernet);
NXLIB_ITEM(Adapters);
// NXLIB_ITEM(IpSubnetMask);
// NXLIB_ITEM(IpAddress);
NXLIB_ITEM(Active);
NXLIB_ITEM(Connected);
NXLIB_ITEM(IpBroadcast);
// NXLIB_ITEM(Status)
// NXLIB_ITEM(Active);
// NXLIB_ITEM(Connected);
// NXLIB_ITEM(ValidIpAddress);
NXLIB_ITEM(CableConnected);

NXLIB_ITEM(AveragePoseError);
NXLIB_ITEM(MaxPoseError);
NXLIB_ITEM(RelativeAveragePoseError);
NXLIB_ITEM(RelativeMaxPoseError);
NXLIB_ITEM(PatternVolume);
NXLIB_ITEM(ReprojectionError);
NXLIB_ITEM(EpipolarError);
NXLIB_ITEM(Contrast);
NXLIB_ITEM(Pattern);
NXLIB_ITEM(Points);
NXLIB_ITEM(ObjectPoints);

NXLIB_ITEM(Threads);
NXLIB_ITEM(CUDA);
// NXLIB_ITEM(Available);
NXLIB_ITEM(Enabled);
NXLIB_ITEM(UseFloat16);
NXLIB_ITEM(Device);
NXLIB_ITEM(Devices);
// NXLIB_ITEM(Name);
NXLIB_ITEM(ComputeCapability);
NXLIB_ITEM(Memory);
NXLIB_ITEM(Cores);
NXLIB_ITEM(ClockRate);
NXLIB_ITEM(Integrated);
NXLIB_ITEM(Capture);
NXLIB_ITEM(SurfaceConnectivity);
NXLIB_ITEM(UseOpenGL);
// NXLIB_ITEM(UEye);
NXLIB_ITEM(OpenMP);
NXLIB_ITEM(EthernetConfigMode);
NXLIB_ITEM(ComputeDisparityMap);
NXLIB_ITEM(StaticBuffers);
NXLIB_ITEM(Physics);
NXLIB_ITEM(Gravity);
NXLIB_ITEM(GroundPlane);
NXLIB_ITEM(RenderView);
NXLIB_ITEM(ShowSurface);
NXLIB_ITEM(ShowCameras);
NXLIB_ITEM(ShowGrid);
NXLIB_ITEM(ShowUserDefinedModels);
NXLIB_ITEM(RenderDepthMap); // Deprecated since 1.2, renamed to RenderPointMap.
NXLIB_ITEM(RenderPointMap);
NXLIB_ITEM(RenderPointMapTexture);
NXLIB_ITEM(PointMap);
NXLIB_ITEM(Normals);
NXLIB_ITEM(DisparityMap);
NXLIB_ITEM(Links);
NXLIB_ITEM(Objects);
// NXLIB_ITEM(Filename);
// NXLIB_ITEM(Pose);
// NXLIB_ITEM(Link);
// NXLIB_ITEM(Target);
// NXLIB_ITEM(DisparityMap);
NXLIB_ITEM(StereoMatching);
NXLIB_ITEM(MinDisparity); // Hidden link to MinimumDisparity (renamed from Version 1.0 to 1.1)
NXLIB_ITEM(MinimumDisparity);
NXLIB_ITEM(NumberOfDisparities);
NXLIB_ITEM(ScaledMinimumDisparity);
NXLIB_ITEM(ScaledNumberOfDisparities);
NXLIB_ITEM(DepthChangeCost);
NXLIB_ITEM(DepthStepCost);
NXLIB_ITEM(WindowRadius);
NXLIB_ITEM(PropagationDecay);
NXLIB_ITEM(CostScale);
NXLIB_ITEM(OptimizationProfile); // deprecated, replaced by Method
// NXLIB_ITEM(Method);
NXLIB_ITEM(ShadowingThreshold);
NXLIB_ITEM(Padding);
NXLIB_ITEM(PhaseInterpolation);
NXLIB_ITEM(PostProcessing);
NXLIB_ITEM(UniquenessRatio);
NXLIB_ITEM(UniquenessOffset);
NXLIB_ITEM(SpeckleRemoval);
NXLIB_ITEM(RegionSize);
NXLIB_ITEM(ComponentThreshold);
NXLIB_ITEM(Filling);
// NXLIB_ITEM(RegionSize);
NXLIB_ITEM(BorderSpread);
NXLIB_ITEM(MedianFilterRadius);
NXLIB_ITEM(RegionFilterDownsampling);
NXLIB_ITEM(MeasurementVolume);
NXLIB_ITEM(Near);
NXLIB_ITEM(Far);
NXLIB_ITEM(DisparityStep);
NXLIB_ITEM(LeftTop);
NXLIB_ITEM(RightTop);
NXLIB_ITEM(RightBottom);
NXLIB_ITEM(LeftBottom);
NXLIB_ITEM(ScaledMeasurementVolume);
NXLIB_ITEM(ValidRegion);
NXLIB_ITEM(AreaOfInterest);
NXLIB_ITEM(ScaledAreaOfInterest);

NXLIB_ITEM(Overlay);
NXLIB_ITEM(Font); // Deprecated
                  // NXLIB_ITEM(Text);
// NXLIB_ITEM(Color);
// NXLIB_ITEM(Angle);
NXLIB_ITEM(Mirror);
NXLIB_ITEM(Vertical);
NXLIB_ITEM(Horizontal);

NXLIB_ITEM(Images);
NXLIB_ITEM(Raw);
NXLIB_ITEM(WithOverlay);
NXLIB_ITEM(Rectified);
NXLIB_ITEM(Left);
NXLIB_ITEM(Right);

NXLIB_ITEM(PatternCount);
NXLIB_ITEM(StereoPatternCount);
NXLIB_ITEM(MonocularPatternCount);
NXLIB_ITEM(Background);
NXLIB_ITEM(StereoCalibrationOnly);

NXLIB_ITEM(Check);

NXLIB_ITEM(ShowPatterns);
NXLIB_ITEM(ShowPatternPoints);
NXLIB_ITEM(ShowObjectPoints);
NXLIB_ITEM(All);
NXLIB_ITEM(Filter);
// NXLIB_ITEM(And);
// NXLIB_ITEM(Or);
// NXLIB_ITEM(Reduce);
// NXLIB_ITEM(UseModel);
// NXLIB_ITEM(Invert);

NXLIB_ITEM(TiltDirection);
NXLIB_ITEM(HUD);
NXLIB_ITEM(Filters);
NXLIB_ITEM(Mask);

NXLIB_ITEM(CalTabType);
// NXLIB_VALUE(Static);
// NXLIB_VALUE(GridSpacing);
// NXLIB_VALUE(TiltDirection);
// NXLIB_VALUE(Mask);
// NXLIB_VALUE(GridWidth);
// NXLIB_VALUE(GridHeight);
// NXLIB_VALUE(Type);
// NXLIB_VALUE(SerialNumber);
// NXLIB_VALUE(Index);

NXLIB_ITEM(Errors);
NXLIB_ITEM(Commands);
NXLIB_ITEM(Items);
// NXLIB_ITEM(Values);
NXLIB_ITEM(ApiErrors);
NXLIB_ITEM(ItemTypes);
NXLIB_ITEM(CommonItems);
NXLIB_ITEM(CommonValues);

NXLIB_VALUE(Available);
NXLIB_VALUE(InUse);
NXLIB_VALUE(Open);
NXLIB_VALUE(Successful);

NXLIB_VALUE(Triggered);
NXLIB_VALUE(Untriggered);

NXLIB_VALUE(New);
NXLIB_VALUE(Add);

NXLIB_VALUE(Top);
NXLIB_VALUE(Bottom);

NXLIB_VALUE(Stereo);
// NXLIB_VALUE(Monocular);
NXLIB_VALUE(Projector);
NXLIB_VALUE(Item);

NXLIB_VALUE(Workspace);
NXLIB_VALUE(Hand);

NXLIB_VALUE(Origin);
NXLIB_VALUE(Axis);

NXLIB_VALUE(X);
NXLIB_VALUE(Y);
NXLIB_VALUE(Z);

NXLIB_VALUE(Float);
NXLIB_VALUE(Validate);

// all deprecated, replaced by constants with Sgm prefix
NXLIB_VALUE(Aligned);
NXLIB_VALUE(Diagonal);
NXLIB_VALUE(AlignedAndDiagonal);

NXLIB_VALUE(Console);
NXLIB_VALUE(Buffer);
NXLIB_VALUE(DebugOut);

NXLIB_VALUE(Plane);
// NXLIB_VALUE(Sphere);
// NXLIB_VALUE(Cylinder);

NXLIB_VALUE(Euclidean);
NXLIB_VALUE(ReprojectionError);

NXLIB_VALUE(Halcon);
NXLIB_VALUE(Ensenso);

NXLIB_ERROR(PatternNotFound);
NXLIB_ERROR(PatternNotDecodable);
NXLIB_ERROR(PatternDataIncompatible);
NXLIB_ERROR(PatternBufferLocked);
NXLIB_ERROR(CommandUnknown);
NXLIB_ERROR(CommandNotAllowed);
NXLIB_ERROR(UnhandledException);
NXLIB_ERROR(OperationCanceled);
NXLIB_ERROR(InvalidPatternBuffer);
NXLIB_ERROR(NoWorkspaceLink);
NXLIB_ERROR(CalibrationFailed);
NXLIB_ERROR(CameraNotFound);
NXLIB_ERROR(InvalidCalibrationData);
NXLIB_ERROR(InvalidPairingData);
NXLIB_ERROR(NotEnoughPointsForPrimitive);
NXLIB_ERROR(EmptyImage);
NXLIB_ERROR(ChangedModel);
NXLIB_ERROR(WiringTestFailed);
NXLIB_ERROR(SensorNotCompatible);

NXLIB_ERROR_(FlexViewAvailableImageCountMismatch, "FlexView/AvailableImageCountMismatch");

NXLIB_VALUE_(FilePrefix, "file://");
NXLIB_VALUE_(KitModelTag, "Kit");
NXLIB_VALUE_(LogFileExt, ".nxlog");

#include "nxLibApiErrors.h"
#ifdef ENSENSO_DEV
#	include "exceptions/exceptions/commonSymbols.h"
#else
#	include "nxLibSymbols.h"
#endif

// Tree item types

NXLIB_INT_CONST(ItemTypeInvalid, 0);
NXLIB_INT_CONST(ItemTypeNull, 1);
NXLIB_INT_CONST(ItemTypeNumber, 2);
NXLIB_INT_CONST(ItemTypeString, 3);
NXLIB_INT_CONST(ItemTypeBool, 4);
NXLIB_INT_CONST(ItemTypeArray, 5);
NXLIB_INT_CONST(ItemTypeObject, 6);

#endif /* __NXLIB_CONSTANTS_H__ */
