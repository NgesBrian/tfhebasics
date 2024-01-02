
To compile the C-API
    git clone https://github.com/zama-ai/tfhe-rs.git
    cd tfhe-rs
    git switch releases/0.4.x

    RUSTFLAGS="-C target-cpu=native" cargo +nightly build --release --features=x86_64-unix,high-level-c-api,safe-deserialization -p tfhe

create the MakeLists.txt and copy content
change 
    set(TFHE_C_API "./tfhe-rs/target/release")

then
 run the commands on the website. 
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE
    make
    ./my-executable