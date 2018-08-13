## 问题描述：
在用Detectron训练自己的训练集时，遇到了一个训练中断的问题(可以正常开始训练，但是训练一段时间后会报错中断，报错信息类似下面这种)
尝试过调整batch_size大小，最终通过参考链接pfollmann提供的解决方案，修改时注意下文件目录设置(Detectron文件目录结构有过更改)，
使得原本会中断的数据集可以跑完训练。
```
*** Aborted at 1525523656 (unix time) try "date -d @1525523656" if you are using GNU date ***
PC: @     0x7f7c0376048a (unknown)
*** SIGSEGV (@0x0) received by PID 89364 (TID 0x7f79559fb700) from PID 0; stack trace: ***
    @     0x7f7c03abd390 (unknown)
    @     0x7f7c0376048a (unknown)
    @     0x7f7c03763cde (unknown)
    @     0x7f7c03766184 __libc_malloc
    @     0x7f7b7f400a36 (unknown)
    @     0x7f7b7f979634 (unknown)
    @     0x7f7b7fa10d34 (unknown)
    @     0x7f7b7fa131b7 (unknown)
    @     0x7f7b7f409ecb (unknown)
    @     0x7f7b7f40a40c cudnnConvolutionBackwardFilter
    @     0x7f7bc570ac3b _ZN6caffe210CuDNNState7executeIRZNS_19CudnnConvGradientOp13DoRunWithTypeIfffffffEEbvEUlPS0_E1_EEvP11CUstream_stOT_
    @     0x7f7bc571337c caffe2::CudnnConvGradientOp::DoRunWithType<>()
    @     0x7f7bc56fead0 caffe2::CudnnConvGradientOp::RunOnDevice()
    @     0x7f7bc568694b caffe2::Operator<>::Run()
    @     0x7f7bf797ec5a caffe2::DAGNet::RunAt()
    @     0x7f7bf797da15 caffe2::DAGNetBase::WorkerFunction()
    @     0x7f7bfd6b7c80 (unknown)
    @     0x7f7c03ab36ba start_thread
    @     0x7f7c037e941d clone
    @                0x0 (unknown)
```

## 参考：
https://github.com/facebookresearch/Detectron/issues/415  pfollmann提供的解决方案。

## Bug修复步骤如下:
1. https://github.com/jcrudy/cython-argsort/blob/master/cyargsort/argsort.pyx. 下载argsort.pyx该函数。
2. 将argsort.pyx放置在detectron/utils目录下。
3. 修改argsort.pyx第13行为:
```
ctypedef cnp.float32_t FLOAT_t
```
4. 在setup.py文件下添加argsort.pyx的信息(模仿cython_nms.pyx and cython_bbox.pyx修改)：
```
ext_modules = [
    Extension(
        name='detectron.utils.cython_bbox',
        sources=[
            'detectron/utils/cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
    Extension(
        name='detectron.utils.argsort',
        sources=[
            'detectron/utils/argsort.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )
    ...
]
```
5. 修改detectron/utils/cython_nms.pyx文件两处：
(1)
```
cimport cython
import numpy as np
cimport numpy as np
# 此处添加一行
import detectron.utils.argsort as argsort
```
(2)注意添加代码的位置（此处掉坑里很久，注意并不是在原来位置替换！）
```
cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
# 注释掉原来的这行
# cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

cdef int ndets = dets.shape[0]
cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros((ndets), dtype=np.int)

# 在此添加新的代码，因为此处用到了上面定义的ndets，如果在原来的地方替换，编译时会报错，无法编译成功。
cdef np.ndarray[np.int_t, ndim=1] order = np.empty((ndets), dtype=np.intp)
argsort.argsort(-scores, order)
```
6. 上述步骤都准确无误的修改完后，在detectron目录下输入"make"重新编译cython模块即可。
