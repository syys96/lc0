rm -r build
meson build --buildtype release -Dopenblas_include="/usr/local/opt/openblas/include" \
      -Dopenblas_libdirs="/usr/local/opt/openblas/lib" \
      -Dblas=true -Dgtest=true -Dmkl=false -Dopencl=false
cd build
ninja
