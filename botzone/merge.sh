cd ..
git submodule update --init --recursive
cd botzone
python ../scripts/compile_proto.py ../libs/lczero-common/proto/net.proto \
        --proto_path=../libs/lczero-common --cpp_out=./
/Users/syys/software/anaconda3/bin/python merge.py
cd cmake-build-release
cmake ..
make


