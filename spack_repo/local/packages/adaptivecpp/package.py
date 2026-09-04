# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import json
import os

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


class Adaptivecpp(CMakePackage):
    """Compiler for multiple programming models (SYCL, C++ standard parallelism,
    HIP/CUDA) for CPUs and GPUs from all vendors: The independent,
    community-driven compiler for C++-based heterogeneous programming models.
    Lets applications adapt themselves to all the hardware in the system - even
    at runtime!"""

    homepage = "https://github.com/AdaptiveCpp/AdaptiveCpp"
    url = "https://github.com/AdaptiveCpp/AdaptiveCpp/archive/refs/tags/v25.10.0.tar.gz"
    git = "https://github.com/AdaptiveCpp/AdaptiveCpp.git"

    provides("sycl")

    license("BSD-2-Clause")

    version("25.10.0", sha256="334b16ebff373bd2841f83332c2ae9a45ec192f2cf964d5fdfe94e1140776059")
    version("25.02.0", sha256="8cc8a3be7bb38f88d7fd51597e0ec924b124d4233f64da62a31b9945b55612ca")
    version("24.10.0", sha256="3bcd94eee41adea3ccc58390498ec9fd30e1548af5330a319be8ce3e034a6a0b")
    version("24.06.0", sha256="cfa117722fd50295de8b9e1d374a0de0aa2407a47439907972e8e3d9795aa285")
    version("24.02.0", sha256="180bdcbf40db9907ba5b3da06a57e779e1527c62528211f72e9d36a5e46b0956")
    version("23.10.0", sha256="9ac3567c048a848f4e6eadb15b09750357ae399896e802b2f1dcaecf8a090064")
    # version("23.10.0-alpha", sha256="2ea77c3b35686f6bcd07cd068a7671635b2b94d8de7e98a4163adbf534cfa33e")
    version("0.9.4", sha256="d9269c814f5e07b54a58bcef177950f222e22127c8399edc2e627d6b9e250763")
    version("0.9.3", sha256="6a2e2d81bd21209ad0726d5aa377321e177fecb775ad93078259835be0931f51")
    version("0.9.2", sha256="4b2308eb19b978a8528d55fe8c9fbb18d5be51aa0dd1a18a068946d8ddedebb1")

    variant("cuda", default=False, description="Enable CUDA backend for SYCL kernels")
    variant("rocm", default=False, description="Enable ROCM backend for SYCL kernels")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("cmake@3.5:", type="build")
    depends_on("boost +filesystem", when="@:0.8")
    depends_on("boost@1.67.0:1.69.0 +filesystem +fiber +context cxxstd=17", when="@0.9.1:")
    depends_on("python@3:")
    depends_on("llvm@8: +clang", when="~cuda")
    depends_on("llvm@9: +clang", when="+cuda")

    # hipSYCL 0.8.0 supported only LLVM 8-10:
    # (https://github.com/AdaptiveCpp/AdaptiveCpp/blob/v0.8.0/CMakeLists.txt#L29-L37)
    # recent versions support only up to llvm18
    # https://github.com/spack/spack/issues/46681
    # https://github.com/spack/spack/issues/49506

    # The following list was made based on the version tested in adaptivecpp github
    depends_on("llvm@15:20", when="@develop")
    depends_on("llvm@15:20", when="@stable")

    depends_on("llvm@15:20", when="@25.10.0")
    depends_on("llvm@15:20", when="@25.02.0")
    depends_on("llvm@14:18", when="@24.10.0")
    depends_on("llvm@14:18", when="@24.06.0")
    depends_on("llvm@13:17", when="@24.02.0")
    depends_on("llvm@13:17", when="@23.10.0")
    depends_on("llvm@11:15", when="@0.9.4")
    depends_on("llvm@11:14", when="@0.9.3")
    depends_on("llvm@11:13", when="@0.9.2")
    depends_on("llvm@11", when="@0.9.1")
    # depends_on("llvm@10:11", when="@0.9.0") # missing in releases
    depends_on("llvm@8:10", when="@0.8.0")

    # https://github.com/spack/spack/issues/45029 and https://github.com/spack/spack/issues/43142
    conflicts("^gcc@12", when="@23.10.0")
    # https://github.com/OpenSYCL/OpenSYCL/pull/918 was introduced after 0.9.4
    conflicts("^gcc@12.2.0", when="@:0.9.4")
    # LLVM PTX backend requires cuda7:10.1 (https://tinyurl.com/v82k5qq)
    depends_on("cuda@9:10.1", when="@0.8.1: +cuda ^llvm@9")
    depends_on("cuda@9:", when="@0.8.1: +cuda ^llvm@10:")
    # hipSYCL@:0.8.0 requires cuda@9:10.0 due to a known bug
    depends_on("cuda@9:10.0", when="@:0.8.0 +cuda")

    conflicts(
        "%gcc@:4",
        when="@:0.9.0",
        msg="hipSYCL needs proper C++14 support to be built, %gcc is too old",
    )
    conflicts(
        "%gcc@:8",
        when="@0.9.1:",
        msg="hipSYCL needs proper C++17 support to be built, %gcc is too old",
    )
    conflicts(
        "^llvm build_type=Debug",
        when="+cuda",
        msg="LLVM debug builds don't work with hipSYCL CUDA backend; for "
        "further info please refer to: "
        "https://github.com/illuhad/hipSYCL/blob/master/doc/install-cuda.md",
    )

    def cmake_args(self):
        spec = self.spec
        args = [
            "-DWITH_CPU_BACKEND:Bool=TRUE",
            "-DWITH_ROCM_BACKEND:Bool={0}".format("TRUE" if spec.satisfies("+rocm") else "FALSE"),
            "-DWITH_CUDA_BACKEND:Bool={0}".format("TRUE" if spec.satisfies("+cuda") else "FALSE"),
            # prevent hipSYCL's cmake to look for other LLVM installations
            # if the specified one isn't compatible
            "-DDISABLE_LLVM_VERSION_CHECK:Bool=TRUE",
        ]
        # LLVM directory containing all installed CMake files
        # (e.g.: configs consumed by client projects)
        llvm_cmake_dirs = find(spec["llvm"].prefix, "LLVMExports.cmake")
        if len(llvm_cmake_dirs) != 1:
            raise InstallError(
                "concretized llvm dependency must provide "
                "a unique directory containing CMake client "
                "files, found: {0}".format(llvm_cmake_dirs)
            )
        args.append("-DLLVM_DIR:String={0}".format(os.path.dirname(llvm_cmake_dirs[0])))
        # clang internal headers directory
        llvm_clang_include_dirs = find(spec["llvm"].prefix, "__clang_cuda_runtime_wrapper.h")
        if len(llvm_clang_include_dirs) != 1:
            raise InstallError(
                "concretized llvm dependency must provide a "
                "unique directory containing clang internal "
                "headers, found: {0}".format(llvm_clang_include_dirs)
            )
        args.append(
            "-DCLANG_INCLUDE_PATH:String={0}".format(os.path.dirname(llvm_clang_include_dirs[0]))
        )
        # target clang++ executable
        llvm_clang_bin = os.path.join(spec["llvm"].prefix.bin, "clang++")
        if not is_exe(llvm_clang_bin):
            raise InstallError(
                "concretized llvm dependency must provide a "
                "valid clang++ executable, found invalid: "
                "{0}".format(llvm_clang_bin)
            )
        args.append("-DCLANG_EXECUTABLE_PATH:String={0}".format(llvm_clang_bin))
        # explicit CUDA toolkit
        if spec.satisfies("+cuda"):
            args.append("-DCUDA_TOOLKIT_ROOT_DIR:String={0}".format(spec["cuda"].prefix))
        if spec.satisfies("+rocm"):
            args.append("-DWITH_ACCELERATED_CPU:STRING=OFF")
            args.append("-DROCM_PATH:STRING={0}".format(os.environ.get("ROCM_PATH")))
            if self.spec.satisfies("@24.02.0:"):
                args.append("-DWITH_SSCP_COMPILER=OFF")
        return args

    @run_after("install")
    def filter_config_file(self):
        def edit_config(filename, editor):
            config_file_paths = find(self.prefix, filename)
            if len(config_file_paths) != 1:
                raise InstallError(
                    "installed hipSYCL must provide a unique compiler driver"
                    "configuration file ({0}), found: {1}".format(filename, config_file_paths)
                )
            config_file_path = config_file_paths[0]
            with open(config_file_path) as f:
                config = json.load(f)

            config_modified = editor(config)

            with open(config_file_path, "w") as f:
                json.dump(config_modified, f, indent=2)

        if self.spec.satisfies("@:23.10.0"):
            configfiles = {"core": "syclcc.json", "cuda": "syclcc.json"}
        else:
            configfiles = {"core": "acpp-core.json", "cuda": "acpp-cuda.json"}

        def adjust_core_config(config):
            config["default-cpu-cxx"] = self.compiler.cxx
            return config

        edit_config(configfiles["core"], adjust_core_config)

        if self.spec.satisfies("+cuda"):
            # 1. Fix compiler: use the real one in place of the Spack wrapper

            # 2. Fix stdlib: we need to make sure cuda-enabled binaries find
            #    the libc++.so and libc++abi.so dyn linked to the sycl
            #    ptx backend
            rpaths = set()
            if self.spec.satisfies("~rocm"):
                so_paths = find_libraries(
                    "libc++", self.spec["llvm"].prefix, shared=True, recursive=True
                )
                if len(so_paths) != 1:
                    raise InstallError(
                        "concretized llvm dependency must provide a "
                        "unique directory containing libc++.so, "
                        "found: {0}".format(so_paths)
                    )
                rpaths.add(os.path.dirname(so_paths[0]))
                so_paths = find_libraries(
                    "libc++abi", self.spec["llvm"].prefix, shared=True, recursive=True
                )
                if len(so_paths) != 1:
                    raise InstallError(
                        "concretized llvm dependency must provide a "
                        "unique directory containing libc++abi, "
                        "found: {0}".format(so_paths)
                    )
                rpaths.add(os.path.dirname(so_paths[0]))

                def adjust_cuda_config(config):
                    config["default-cuda-link-line"] += " " + " ".join(
                        "-rpath {0}".format(p) for p in rpaths
                    )
                    return config

                edit_config(configfiles["cuda"], adjust_cuda_config)