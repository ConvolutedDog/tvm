# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import ctypes
import traceback
from cpython cimport Py_INCREF, Py_DECREF, PyGILState_Ensure, PyGILState_Release
from numbers import Number, Integral
from ..base import string_types, py2cerror
from ..runtime_ctypes import DataType, Device, TVMByteArray, ObjectRValueRef


cdef void tvm_callback_finalize(void* fhandle) with gil:
    """The finalizer on resource handle (2nd param of TVMFuncCreateFromCFunc,
    which wraps a TVMPackedCFunc to become a FunctionHandle) when the
    FunctionHandle get freed, can be NULL. In fact it is used as the deleter
    of the fhandle in backend C++.
    """
    local_pyfunc = <object>(fhandle)
    Py_DECREF(local_pyfunc)

cdef int tvm_callback(TVMValue* args,
                      int* type_codes,
                      int num_args,
                      TVMRetValueHandle ret,
                      void* fhandle) with gil:
    """The tvm_callback function is the 1st param of TVMFuncCreateFromCFunc.
    And it is the callback function to be called by TVM backend C++ to define
    a PackedFunc using passed-in arguments from frontend Python. In fact, the
    registered lambda function will invoke tvm_callback to do the work:

    1. fhandle is the Python func handle as the last argument of tvm_callback.
    2. TVMValue* args, int* type_codes, int num_args, TVMRetValueHandle ret,
       these three arguments should be provided when invoking the registered
       function.
    3. This function will be called by TVM backend C++ to invoke and execute
       the registered Python function.
    4. If the local Python function returns a value (rv is not None), this
       function should be stored in ret, because once the local Python func
       is invoked from the C++ backend side, the return value will be only
       returned to the frontend Python side (after all, Python functions are
       executed on the Python side). So, at this point, we need to copy the
       return value to ret.

    Its signature is:

    typedef int (*TVMPackedCFunc)(TVMValue* args,
                                  int* type_codes,
                                  int num_args,
                                  TVMRetValueHandle ret,
                                  void* resource_handle);

    \brief C type of packed function.

    \param args The arguments
    \param type_codes The type codes of the arguments
    \param num_args Number of arguments.
    \param ret The return value handle.
    \param resource_handle The handle additional resouce handle from front-end.
    \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
    \sa TVMCFuncSetReturn
    """
    cdef list pyargs
    cdef TVMValue value
    cdef int tcode
    local_pyfunc = <object>(fhandle)
    pyargs = []
    for i in range(num_args):
        value = args[i]
        tcode = type_codes[i]
        if (tcode == kTVMObjectHandle or
            tcode == kTVMPackedFuncHandle or
            tcode == kTVMModuleHandle or
            tcode == kTVMNDArrayHandle or
            tcode == kTVMObjectRefArg or
            tcode >= kTVMExtBegin):
            # If the argument is handle, we need to convert it to TVMRetValue,
            # as the content represented by this handle may be modified by the
            # backend C++ and will be former use in the frontend Python.
            #
            # \brief Inplace translate callback argument value to return value.
            # This is only needed for non-POD arguments.
            #
            # \param value The value to be translated.
            # \param code The type code to be translated.
            # \note This function will do a shallow copy when necessary.
            #
            # \return 0 when success, nonzero when failure happens.
            CHECK_CALL(TVMCbArgToReturn(&value, &tcode))

        if tcode != kTVMDLTensorHandle:
            pyargs.append(make_ret(value, tcode))
        else:
            pyargs.append(c_make_array(value.v_handle, True, False))
    try:
        # Execute the frontend Python function, this is the reason why backend
        # C++ can call frontend Python function easily.
        rv = local_pyfunc(*pyargs)
    except Exception as err:
        msg = traceback.format_exc()
        msg = py2cerror(msg)
        TVMAPISetLastPythonError(<void*>err)

        return -1
    if rv is not None:
        if isinstance(rv, tuple):
            raise ValueError("PackedFunction can only support one return value")
        temp_args = []
        # Pack arguments into c args tvm call accept
        make_arg(rv, &value, &tcode, temp_args)

        # \brief Set the return value of TVMPackedCFunc.
        #
        # TVMCFuncSetReturn is called by TVMPackedCFunc to set the return value.
        # When this function is not called, the function returns null by default.
        #
        # \param ret The return value handle, pass by ret in TVMPackedCFunc.
        # \param value The value to be returned.
        # \param type_code The type of the value to be returned.
        # \param num_ret Number of return values, for now only 1 is supported.
        CHECK_CALL(TVMCFuncSetReturn(ret, &value, &tcode, 1))
    return 0


cdef object make_packed_func(TVMPackedFuncHandle chandle, int is_global):
    """After convert_to_tvm_func is called, this function will be called to
    create the python object according to the PackedFunc function address
    (chandle) returned by the backend C++.
    """
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    (<PackedFuncBase>obj).chandle = chandle
    (<PackedFuncBase>obj).is_global = is_global
    return obj


def convert_to_tvm_func(object pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.Function
        The converted tvm function.
    """
    cdef TVMPackedFuncHandle chandle
    Py_INCREF(pyfunc)

    # TVMFuncCreateFromCFunc is defined in include/tvm/runtime/c_runtime_api.h
    #
    # TVM_DLL int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
    #                                    void* resource_handle,
    #                                    TVMPackedCFuncFinalizer fin,
    #                                    TVMFunctionHandle* out);
    #
    # \brief Wrap a TVMPackedCFunc to become a FunctionHandle.
    #
    # The resource_handle will be managed by TVM API, until the function
    # is no longer used.
    #
    # \param func The packed C function.
    # \param resource_handle The resource handle from front-end,
    #                        can be NULL.
    # \param fin The finalizer on resource handle when the FunctionHandle
    #           get freed, can be NULL.
    # \param out the result function handle.
    # \return 0 when success, nonzero when failure happens.
    #
    # In fact, in the backend, this function mainly does the following:
    #
    # \code{.cpp}
    # ret = PackedFunc([func, resource_handle](TVMArgs args, TVMRetValue* rv) {
    #   int ret = func(const_cast<TVMValue*>(args.values), const_cast<int*>(args.type_codes),
    #                  args.num_args, rv, resource_handle);
    #   if (ret != 0) {
    #     TVMThrowLastError();
    #   }
    # });
    # // MoveToCHost moves the value back to front-end via C API.
    # ret.MoveToCHost(&val, &type_code);
    # *out = val.v_handle;
    # \endcode
    #
    # So that here, chandle is actually a PackedFunc object's address to
    # the packed function.
    CHECK_CALL(TVMFuncCreateFromCFunc(tvm_callback,
                                      <void*>(pyfunc),
                                      tvm_callback_finalize,
                                      &chandle))
    # Create the python object according to the PackedFunc function address
    # (chandle) returned by the backend C++.
    return make_packed_func(chandle, False)


cdef inline int make_arg(object arg,
                         TVMValue* value,
                         int* tcode,
                         list temp_args) except -1:
    """Pack arguments into c args tvm call accept"""
    cdef unsigned long long ptr
    if isinstance(arg, ObjectBase):
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kTVMObjectHandle
    elif isinstance(arg, NDArrayBase):
        value[0].v_handle = (<NDArrayBase>arg).chandle
        tcode[0] = (kTVMNDArrayHandle if
                    not (<NDArrayBase>arg).c_is_view else kTVMDLTensorHandle)
    elif isinstance(arg, PyNativeObject):
        value[0].v_handle = (<ObjectBase>(arg.__tvm_object__)).chandle
        tcode[0] = kTVMObjectHandle
    elif isinstance(arg, _TVM_COMPATS):
        ptr = arg._tvm_handle
        value[0].v_handle = (<void*>ptr)
        tcode[0] = arg.__class__._tvm_tcode
    elif isinstance(arg, bool):
        # A python `bool` is a subclass of `int`, so this check
        # must occur before `Integral`.
        value[0].v_int64 = arg
        tcode[0] = kTVMArgBool
    elif isinstance(arg, Integral):
        value[0].v_int64 = arg
        tcode[0] = kInt
    elif isinstance(arg, float):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, str):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif arg is None:
        value[0].v_handle = NULL
        tcode[0] = kTVMNullptr
    elif isinstance(arg, Number):
        value[0].v_float64 = arg
        tcode[0] = kFloat
    elif isinstance(arg, DataType):
        tstr = c_str(str(arg))
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif isinstance(arg, Device):
        value[0].v_device = (<DLDevice*>(
            <unsigned long long>ctypes.addressof(arg)))[0]
        tcode[0] = kDLDevice
    elif isinstance(arg, (bytes, bytearray)):
        # from_buffer only taeks in bytearray.
        if isinstance(arg, bytes):
            byte_arr = bytearray(arg)
            temp_args.append(byte_arr)
            arg = byte_arr

        arr = TVMByteArray()
        arr.data = ctypes.cast(
            (ctypes.c_byte * len(arg)).from_buffer(arg),
            ctypes.POINTER(ctypes.c_byte))
        arr.size = len(arg)
        value[0].v_handle = <void*>(
            <unsigned long long>ctypes.addressof(arr))
        tcode[0] = kTVMBytes
        temp_args.append(arr)
    elif isinstance(arg, string_types):
        tstr = c_str(arg)
        value[0].v_str = tstr
        tcode[0] = kTVMStr
        temp_args.append(tstr)
    elif isinstance(arg, (list, tuple, dict, _CLASS_OBJECT_GENERIC)):
        arg = _FUNC_CONVERT_TO_OBJECT(arg)
        value[0].v_handle = (<ObjectBase>arg).chandle
        tcode[0] = kTVMObjectHandle
        temp_args.append(arg)
    elif isinstance(arg, _CLASS_MODULE):
        value[0].v_handle = c_handle(arg.handle)
        tcode[0] = kTVMModuleHandle
    elif isinstance(arg, PackedFuncBase):
        value[0].v_handle = (<PackedFuncBase>arg).chandle
        tcode[0] = kTVMPackedFuncHandle
    elif isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kTVMOpaqueHandle
    elif isinstance(arg, ObjectRValueRef):
        value[0].v_handle = &((<ObjectBase>(arg.obj)).chandle)
        tcode[0] = kTVMObjectRefArg
    elif callable(arg):
        arg = convert_to_tvm_func(arg)
        value[0].v_handle = (<PackedFuncBase>arg).chandle
        tcode[0] = kTVMPackedFuncHandle
        temp_args.append(arg)
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))
    return 0


cdef inline bytearray make_ret_bytes(void* chandle):
    handle = ctypes_handle(chandle)
    arr = ctypes.cast(handle, ctypes.POINTER(TVMByteArray))[0]
    size = arr.size
    res = bytearray(size)
    rptr = (ctypes.c_byte * size).from_buffer(res)
    if not ctypes.memmove(rptr, arr.data, size):
        raise RuntimeError('memmove failed')
    return res


cdef inline object make_ret(TVMValue value, int tcode):
    """convert result to return value."""
    if tcode == kTVMObjectHandle:
        return make_ret_object(value.v_handle)
    elif tcode == kTVMNullptr:
        return None
    elif tcode == kTVMArgBool:
        return bool(value.v_int64)
    elif tcode == kInt:
        return value.v_int64
    elif tcode == kFloat:
        return value.v_float64
    elif tcode == kTVMNDArrayHandle:
        return c_make_array(value.v_handle, False, True)
    elif tcode == kTVMStr:
        return py_str(value.v_str)
    elif tcode == kTVMBytes:
        return make_ret_bytes(value.v_handle)
    elif tcode == kTVMOpaqueHandle:
        return ctypes_handle(value.v_handle)
    elif tcode == kDLDevice:
        return Device(value.v_device.device_type, value.v_device.device_id)
    elif tcode == kTVMModuleHandle:
        return _CLASS_MODULE(ctypes_handle(value.v_handle))
    elif tcode == kTVMPackedFuncHandle:
        return make_packed_func(value.v_handle, False)
    elif tcode in _TVM_EXT_RET:
        return _TVM_EXT_RET[tcode](ctypes_handle(value.v_handle))

    raise ValueError("Unhandled type code %d" % tcode)


cdef inline int FuncCall3(void* chandle,
                          tuple args,
                          int nargs,
                          TVMValue* ret_val,
                          int* ret_tcode) except -1:
    cdef TVMValue[3] values
    cdef int[3] tcodes
    nargs = len(args)
    temp_args = []
    # Pack arguments into c args tvm call accept
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)

    with nogil:
        c_api_ret_code = TVMFuncCall(chandle, &values[0], &tcodes[0],
                                     nargs, ret_val, ret_tcode)

    CHECK_CALL(c_api_ret_code)
    return 0

cdef inline int FuncCall(void* chandle,
                         tuple args,
                         TVMValue* ret_val,
                         int* ret_tcode) except -1:
    """Invoke the registered function in the function table in
    backend C++.
    """
    cdef int nargs
    cdef int c_api_ret_code
    nargs = len(args)
    # If the number of args is less than or qual to 3, just use
    # an array to store the arguments. Else, use a vector to store
    # the arguments.
    if nargs <= 3:
        FuncCall3(chandle, args, nargs, ret_val, ret_tcode)
        return 0

    cdef vector[TVMValue] values
    cdef vector[int] tcodes
    values.resize(max(nargs, 1))
    tcodes.resize(max(nargs, 1))
    temp_args = []
    # Pack arguments into c args tvm call accept
    for i in range(nargs):
        make_arg(args[i], &values[i], &tcodes[i], temp_args)

    with nogil:
        c_api_ret_code = TVMFuncCall(chandle, &values[0], &tcodes[0],
                                     nargs, ret_val, ret_tcode)
    CHECK_CALL(c_api_ret_code)
    return 0


cdef inline int ConstructorCall(void* constructor_handle,
                                int type_code,
                                tuple args,
                                void** handle) except -1:
    """Call contructor of a handle function

    For example, TVM defined the following API in python/tvm/arith/_ffi_api.py:

    \code{.py}
    import tvm._ffi
    tvm._ffi._init_api("arith", __name__) # __name__ = tvm.arith
    \endcode

    And the definition of _init_api is as follows:

    \code{.py}
    def _get_api(f):
        flocal = f
        flocal.is_global = True
        return flocal

    def _init_api(namespace, target_module_name=None):
        # Initialize api for a given module name
        #
        # namespace : str
        #    The namespace of the source registry
        #
        # target_module_name : str
        #    The target module name if different from namespace
        #
        target_module_name = 
            target_module_name if target_module_name else namespace
        if namespace.startswith("tvm."):
            _init_api_prefix(target_module_name, namespace[4:])
        else:
        _init_api_prefix(target_module_name, namespace)

    def _init_api_prefix(module_name, prefix):
        module = sys.modules[module_name]

        for name in list_global_func_names():
            if not name.startswith(prefix):
                continue

            fname = name[len(prefix) + 1 :]
            target_module = module

            if fname.find(".") != -1:
                continue
            f = get_global_func(name)
            ff = _get_api(f)
            ff.__name__ = fname
            ff.__doc__ = "TVM PackedFunc %s. " % fname
            setattr(target_module, ff.__name__, ff)
    \endcode

    So, this function actually searches for all global functions that
    start with the prefix ("arith"), and then creates a function with
    the same name in the module (tvm.arith), and assigns the packed
    function to it.

    Then if we call tvm.arith.ConstIntBound(min_val, max_val), it will
    call the underlying packed function in backend C++.

    The following code will call __init_handle_by_constructor__, and
    _ffi_api.ConstIntBound actually invokes the packed function called
    arith.ConstIntBound, which is registered in the backend C++. And
    __init_handle_by_constructor__ will former call ConstructorCall
    and finally call FuncCall, which will invoke the registered func
    in the function table in backend C++.

    \code{.py}
    @tvm._ffi.register_object("arith.ConstIntBound")
    class ConstIntBound(Object):
        POS_INF = (1 << 63) - 1
        NEG_INF = -POS_INF

        def __init__(self, min_value, max_value):
            self.__init_handle_by_constructor__(_ffi_api.ConstIntBound,
                                                min_value, max_value)
    \endcode

    constructor_handle is (<PackedFuncBase>arith.ConstIntBound).chandle,
    type_code is kTVMObjectHandle, args is (min_value, max_value), and
    handle is the frontend arith.ConstIntBound's chandle which is the
    handle to the underlying C++ object.
    """
    cdef TVMValue ret_val
    cdef int ret_tcode
    FuncCall(constructor_handle, args, &ret_val, &ret_tcode)
    assert ret_tcode == type_code
    handle[0] = ret_val.v_handle
    return 0


cdef class PackedFuncBase:
    """This class is corresponding to the PackedFunc in backend C++.
    The chandle is used to hold the handle to the backend PackedFunc,
    and is_global is used to indicate whether the handle is a global
    function.

    Note: chandle is actually a void pointer, and the PackedFuncBase
    class is used to wrap the handle to provide a python interface.
    """
    cdef TVMPackedFuncHandle chandle
    cdef int is_global

    cdef inline _set_handle(self, handle):
        if handle is None:
            self.chandle = NULL
        else:
            self.chandle = c_handle(handle)

    property is_global:
        def __get__(self):
            return self.c_is_global != 0

        def __set__(self, value):
            self.c_is_global = value

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.chandle, ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle, is_global):
        self._set_handle(handle)
        self.c_is_global = is_global

    def __dealloc__(self):
        if self.is_global == 0:
            CHECK_CALL(TVMFuncFree(self.chandle))

    def __call__(self, *args):
        cdef TVMValue ret_val
        cdef int ret_tcode
        ret_tcode = kTVMNullptr
        # Call the registered function in backend C++.
        #
        # Note that the frontend Python function has been registered
        # to the function table in backend C++ as a global function,
        # FuncCall will call TVMFuncCall to invoke the registered
        # Python function.
        FuncCall(self.chandle, args, &ret_val, &ret_tcode)
        return make_ret(ret_val, ret_tcode)


def _get_global_func(name, allow_missing):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : PackedFunc
        The function to be returned, None if function is missing.
    """
    cdef TVMPackedFuncHandle chandle
    CHECK_CALL(TVMFuncGetGlobal(c_str(name), &chandle))
    if chandle != NULL:
        return make_packed_func(chandle, True)

    if allow_missing:
       return None

    raise ValueError("Cannot find global function %s" % name)


_CLASS_PACKED_FUNC = None
_CLASS_MODULE = None
_CLASS_OBJECT = None
_CLASS_OBJECT_GENERIC = None
_FUNC_CONVERT_TO_OBJECT = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_packed_func(func_class):
    global _CLASS_PACKED_FUNC
    _CLASS_PACKED_FUNC = func_class

def _set_class_object(obj_class):
    global _CLASS_OBJECT
    _CLASS_OBJECT = obj_class

def _set_class_object_generic(object_generic_class, func_convert_to_object):
    global _CLASS_OBJECT_GENERIC
    global _FUNC_CONVERT_TO_OBJECT
    _CLASS_OBJECT_GENERIC = object_generic_class
    _FUNC_CONVERT_TO_OBJECT = func_convert_to_object
