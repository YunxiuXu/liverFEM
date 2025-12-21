#ifndef __Version_h__
#define __Version_h__

#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define WARNING(desc) message(__FILE__ "(" STRINGIZE(__LINE__) ") : Warning: " #desc)

#define GIT_SHA1 "c035b62ec395b9807c3d31b048c34f2c0976fb3c"
#define GIT_REFSPEC "refs/heads/main"
#define GIT_LOCAL_STATUS "DIRTY"

#define PBD_VERSION "2.2.2"

#ifdef DL_OUTPUT
#pragma WARNING(Local changes not committed.)
#endif

#endif
