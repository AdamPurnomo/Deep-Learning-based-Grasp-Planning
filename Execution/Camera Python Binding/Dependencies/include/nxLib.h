#ifndef __NXLIB_H__
#define __NXLIB_H__

#include "nxLibConstants.h"
#include "nxLibVersion.h"

#if defined(_WIN32) || defined(__WINDOWS__)
#	if defined(NxLib32_EXPORTS) || defined(NxLib64_EXPORTS)
#		define NXLIB_EXPORT __declspec(dllexport)
#	else
#		define NXLIB_EXPORT __declspec(dllimport)
#	endif
#else /* defined (_WIN32) */
#	define NXLIB_EXPORT
#endif

#ifndef ATTR_UNUSED
#	if defined(_WIN32) || defined(__WINDOWS__)
#		define ATTR_UNUSED
#	else /* defined (_WIN32) */
#		define ATTR_UNUSED __attribute__((unused))
#	endif
#endif

#include "nxLibFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifndef NXLIB_DYNAMIC_LOAD
NXLIB_EXPORT void nxLibInitialize(NXLIBERR* result, NXLIBBOOL waitForInitialCameraRefresh);

NXLIB_EXPORT void nxLibFinalize(NXLIBERR* result);

NXLIB_EXPORT void nxLibOpenTcpPort(NXLIBERR* result, NXLIBINT portNumber, NXLIBINT* openedPort);
NXLIB_EXPORT void nxLibCloseTcpPort(NXLIBERR* result);
#endif
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static void ATTR_UNUSED nxLibInitialize(bool waitForInitialCameraRefresh = true)
{
	int result;
	nxLibInitialize(&result, static_cast<NXLIBBOOL>(waitForInitialCameraRefresh));
	nxLibCheckReturnCode(result);
}

static void ATTR_UNUSED nxLibFinalize()
{
	int result;
	nxLibFinalize(&result);
	nxLibCheckReturnCode(result);
}

static void ATTR_UNUSED nxLibOpenTcpPort(int portNumber = 0, int* openedPort = 0)
{
	int result;
	nxLibOpenTcpPort(&result, portNumber, openedPort);
	nxLibCheckReturnCode(result);
}

static void ATTR_UNUSED nxLibCloseTcpPort()
{
	int result;
	nxLibCloseTcpPort(&result);
	nxLibCheckReturnCode(result);
}
#endif

#endif /*__NXLIB_H_*/
