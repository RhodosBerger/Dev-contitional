#!/bin/bash
set -e

APP_NAME="gamesa-cortex-v2"
VERSION="2.0.0"
ARCH="amd64"
DEB_NAME="${APP_NAME}_${VERSION}_${ARCH}"
BUILD_DIR="build/${DEB_NAME}"

echo "Building ${DEB_NAME}..."

# 1. Clean and Create Build Directory
rm -rf build
mkdir -p "${BUILD_DIR}/usr/local/bin"
mkdir -p "${BUILD_DIR}/usr/lib/${APP_NAME}"
mkdir -p "${BUILD_DIR}/etc/${APP_NAME}"
mkdir -p "${BUILD_DIR}/DEBIAN"

# 2. Copy Source Code
echo "Copying Source..."
cp -r gamesa_cortex_v2/src "${BUILD_DIR}/usr/lib/${APP_NAME}/"
cp -r gamesa_cortex_v2/scripts "${BUILD_DIR}/usr/lib/${APP_NAME}/"
cp gamesa_cortex_v2/README.md "${BUILD_DIR}/usr/lib/${APP_NAME}/"

# 3. Copy Docker Assets
echo "Copying Docker Configs..."
cp Dockerfile "${BUILD_DIR}/usr/lib/${APP_NAME}/"
cp docker-compose.yml "${BUILD_DIR}/usr/lib/${APP_NAME}/"

# 4. Compile Rust (If available)
if command -v cargo &> /dev/null; then
    echo "Compiling Rust Planner..."
    cd gamesa_cortex_v2/rust_planner
    cargo build --release
    cp target/release/librust_planner.so "../../${BUILD_DIR}/usr/lib/${APP_NAME}/src/core/" || echo "Warning: Rust lib copy failed."
    cd ../..
else
    echo "Check: Cargo not found. Skipping Rust compilation (Source only)."
    cp -r gamesa_cortex_v2/rust_planner "${BUILD_DIR}/usr/lib/${APP_NAME}/"
fi

# 5. Create executable wrapper
cat <<EOF > "${BUILD_DIR}/usr/local/bin/${APP_NAME}"
#!/bin/bash
export PYTHONPATH=/usr/lib/${APP_NAME}
python3 -m src.core.npu_coordinator "\$@"
EOF
chmod +x "${BUILD_DIR}/usr/local/bin/${APP_NAME}"

# 6. Create DEBIAN/control
cat <<EOF > "${BUILD_DIR}/DEBIAN/control"
Package: ${APP_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Dusan <dusan@example.com>
Depends: python3 (>= 3.10), python3-numpy, vulkan-tools
Description: Gamesa Cortex V2 - AI NPU & 3D Grid Stack
 Advanced Heterogeneous Computing Stack with Docker support,
 Rust Planning, and Vulkan/OpenCL acceleration for 3D grids.
EOF

# 7. Build Deb
dpkg-deb --build "${BUILD_DIR}"
mv "build/${DEB_NAME}.deb" .
echo "Build Complete: ${DEB_NAME}.deb"
