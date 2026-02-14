#!/bin/bash
set -e

# Version
VERSION="1.6.0"
PKG_NAME="advanced-cnc-copilot-gamesa-grid"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"
PKG_DIR="build/${PKG_NAME}_${VERSION}_${ARCH}"

echo "Building PC Linux Krystal Stack: ${DEB_NAME}..."

# 1. Cleanup
rm -rf build/${PKG_NAME}_${VERSION}_${ARCH}
mkdir -p ${PKG_DIR}/opt/advanced_cnc_copilot
mkdir -p ${PKG_DIR}/DEBIAN

# 2. Copy Codebase
echo "Copying Backend..."
cp -r backend ${PKG_DIR}/opt/advanced_cnc_copilot/
cp main.py ${PKG_DIR}/opt/advanced_cnc_copilot/

if [ -f "requirements.txt" ]; then
    cp requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
elif [ -f "../requirements.txt" ]; then
    cp ../requirements.txt ${PKG_DIR}/opt/advanced_cnc_copilot/
fi

# 3. Include Benchmark Tool
chmod +x ${PKG_DIR}/opt/advanced_cnc_copilot/backend/benchmarks/benchmark_pc_stack.py

# 4. Create Control File
cat > ${PKG_DIR}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Dusan <dusan@example.com>
Description: Advanced CNC Copilot - PC Linux Gamesa/Krystal Stack
  High-Performance GPU Edition.
  Includes OpenVINO Throughput Optimizations and Krystal Benchmark.
EOF

# 5. Build Deb
mkdir -p dist
dpkg-deb --build ${PKG_DIR}
mv build/${DEB_NAME} dist/

echo "PC Stack Build Complete: dist/${DEB_NAME}"
