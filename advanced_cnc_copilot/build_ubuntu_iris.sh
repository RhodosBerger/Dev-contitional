#!/bin/bash
set -e

# Version
VERSION="1.8.0"
PKG_NAME="advanced-cnc-copilot-ubuntu-iris"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"
PKG_DIR="build/${PKG_NAME}_${VERSION}_${ARCH}"

echo "Building Ubuntu Iris Xe Optimized Package: ${DEB_NAME}..."

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

# 3. Include Scripts
cp kernel_tuning.sh ${PKG_DIR}/opt/advanced_cnc_copilot/
cp benchmark_full_stack.sh ${PKG_DIR}/opt/advanced_cnc_copilot/
chmod +x ${PKG_DIR}/opt/advanced_cnc_copilot/*.sh

# 4. Create Control File
cat > ${PKG_DIR}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: Dusan <dusan@example.com>
Description: Advanced CNC Copilot - Iris Xe Edition
  Includes Dynamic Resolution Scaling for Intel Xe Graphics.
  Ubuntu Full Stack Optimization.
EOF

# 5. Build Deb
mkdir -p dist
dpkg-deb --build ${PKG_DIR}
mv build/${DEB_NAME} dist/

echo "Ubuntu Iris Build Complete: dist/${DEB_NAME}"
