

Current Repo Structure:

================================================================================
PURE C++ FILES
main.cpp    - entry point for running rnn application with pure c++
functions.h - holds main functions that have been ported over from matlab .m
              files

================================================================================
MATLAB INTERFACE FILES (MexFunction classes)
modifyidq_ref.cpp/h - Interface class that can be compiled using 'mex' command
                      in matlab to produce executable that can be invoked
                      like any other matlab function.
                      Implements modifyIdq_ref.m

calculateidq_ref.cpp/h - Interface class that can be compiled using 'mex' 
                         command in matlab to produce executable that can be 
                         invoked like any other matlab function.
                         Implements calculateIdq_ref.m

mexfuncbase.h - Base class used for common functionality across all 
                MexFunction interface classes
