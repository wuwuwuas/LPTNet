// pyObjectFinder.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ObjectFinder.h"
#include "ObjectInfo.h"
#include "Config.h"
#include "Matrix.h"

namespace py = pybind11;

void bind_ObjectFinder(py::module_& m) {
    py::class_<ObjectFinder2D>(m, "ObjectFinder2D")
        .def(py::init<>())

        // 直接返回 std::vector<std::unique_ptr<Object2D>>
        .def("findObject2D",
             [](ObjectFinder2D& self, const Image& image, const ObjectConfig& obj_cfg) {
                 return self.findObject2D(image, obj_cfg);
             },
             py::arg("image"), py::arg("obj_cfg"))

        // 这里定义 Python 调用的接口名称，并使用 lambda 来处理参数转换
        .def("findTracer2D_fromNN", 
             [](ObjectFinder2D& self, 
                py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                py::array_t<double, py::array::c_style | py::array::forcecast> diameters,
                py::array_t<double, py::array::c_style | py::array::forcecast> intensities) 
             -> decltype(self.createTracersFromRawData(nullptr, nullptr, nullptr, 0)) // 增加一个 nullptr 占位符
             {
                auto cbuf = centers.request();
                auto dbuf = diameters.request();
                auto ibuf = intensities.request(); // 获取强度 buffer

                if (cbuf.ndim != 2 || cbuf.shape[1] != 2)
                    throw std::runtime_error("centers must be Nx2");
                if (dbuf.ndim != 1)
                    throw std::runtime_error("diameters must be 1D");
                if (ibuf.ndim != 1)
                    throw std::runtime_error("intensities must be 1D"); // 校验强度维度

                size_t N = cbuf.shape[0];
                if (dbuf.shape[0] != N)
                    throw std::runtime_error("size mismatch: centers rows != diameters length");
                if (ibuf.shape[0] != N)
                    throw std::runtime_error("size mismatch: centers rows != intensities length"); // 校验强度长度

                auto* c_ptr = static_cast<double*>(cbuf.ptr);
                auto* d_ptr = static_cast<double*>(dbuf.ptr);
                auto* i_ptr = static_cast<double*>(ibuf.ptr); // 获取强度指针

                return self.createTracersFromRawData(c_ptr, d_ptr, i_ptr, N);
             },
             py::arg("centers"), 
             py::arg("diameters"), // 参数名改为 diameters
             py::arg("intensities"), // 新增 intensities 参数
             py::call_guard<py::gil_scoped_release>()
        );
}




// pyObjectFinder.cpp
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/numpy.h>
// #include "ObjectFinder.h"
// #include "ObjectInfo.h"
// #include "Config.h"
// #include "Matrix.h"
// namespace py = pybind11;

// void bind_ObjectFinder(py::module_& m) {
//     py::class_<ObjectFinder2D>(m, "ObjectFinder2D")
//         .def(py::init<>())

//         // 直接返回 std::vector<std::unique_ptr<Object2D>>
//         .def("findObject2D",
//              [](ObjectFinder2D& self, const Image& image, const ObjectConfig& obj_cfg) {
//                  return self.findObject2D(image, obj_cfg);
//              },
//              py::arg("image"), py::arg("obj_cfg"))

//         // 这里定义 Python 调用的接口名称，并使用 lambda 来处理参数转换
//         .def("findTracer2D_fromNN", 
//              [](ObjectFinder2D& self, 
//                 py::array_t<double, py::array::c_style | py::array::forcecast> centers,
//                 py::array_t<double, py::array::c_style | py::array::forcecast> radii) 
//              -> decltype(self.createTracersFromRawData(nullptr, nullptr, 0)) // 【关键修改】：显式指定返回类型
//              {
//                 // 1. 获取 buffer 信息
//                 auto cbuf = centers.request();
//                 auto rbuf = radii.request();

//                 // 2. 形状与维度安全检查
//                 if (cbuf.ndim != 2 || cbuf.shape[1] != 2)
//                     throw std::runtime_error("centers must be Nx2");
//                 if (rbuf.ndim != 1)
//                     throw std::runtime_error("radii must be 1D");

//                 size_t N = cbuf.shape[0];
//                 if (rbuf.shape[0] != N)
//                     throw std::runtime_error("size mismatch: centers rows != radii length");

//                 // 3. 获取裸指针
//                 auto* c_ptr = static_cast<double*>(cbuf.ptr);
//                 auto* r_ptr = static_cast<double*>(rbuf.ptr);

//                 // 4. 调用纯 C++ 函数
//                 return self.createTracersFromRawData(c_ptr, r_ptr, N);
//              },
//              py::arg("centers"), // python 端参数名
//              py::arg("radii"),
//              py::call_guard<py::gil_scoped_release>() // 【核心修改】：允许其他 Python 线程在此 C++ 运行期间继续执行
//         );
// }
