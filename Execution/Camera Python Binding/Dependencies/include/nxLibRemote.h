#ifndef __NXLIBREMOTE_H__
#define __NXLIBREMOTE_H__

#include "nxLibConstants.h"

#if defined(_WIN32)
#	if defined(NxLibRemote32_EXPORTS) || defined(NxLibRemote64_EXPORTS)
#		define NXLIB_EXPORT __declspec(dllexport)
#	else
#		define NXLIB_EXPORT __declspec(dllimport)
#	endif
#	define ATTR_UNUSED
#else /* defined (_WIN32) */
#	define NXLIB_EXPORT
#	define ATTR_UNUSED __attribute__((unused))
#endif

#include "nxLibFunctions.h"

extern "C" {
NXLIB_EXPORT void nxLibConnect(NXLIBERR* result, NXLIBSTR hostname, NXLIBINT portNumber);
NXLIB_EXPORT void nxLibDisconnect(NXLIBERR* result);
}

#ifdef __cplusplus
static void ATTR_UNUSED nxLibConnect(std::string const& hostname, int port)
{
	int result;
	nxLibConnect(&result, hostname.c_str(), port);
	nxLibCheckReturnCode(result);
}
static void ATTR_UNUSED nxLibDisconnect()
{
	int result;
	nxLibDisconnect(&result);
	nxLibCheckReturnCode(result);
}
#endif /* __cplusplus */

#endif /*__NXLIBREMOTE_H__*/
