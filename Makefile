.PHONY: all clean meson cmake debug dist

all: meson

clean:
	rm -rf build build-meson

VENDOR_FILES=vendor/thread_pool/README.md vendor/spoa/README.md vendor/rampler/README.md vendor/GenomeWorks/README.md

meson: ${VENDOR_FILES}
	@echo "[Invoking Meson]"
	@mkdir -p build-meson && cd build-meson && meson --buildtype=release -Dc_args=-O3 -Dtests=true && ninja

rebuild: ${VENDOR_FILES}
	@echo "[Running Ninja only]"
	@ninja -C build-meson

cmake: ${VENDOR_FILES}
	@echo "[Invoking CMake]"
	@mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -Dracon_build_tests=ON .. && make

debug: ${VENDOR_FILES}
	@echo "[Invoking Meson]"
	@mkdir -p build-debug && cd build-debug && (meson --buildtype=debugoptimized -Db_sanitize=address -Dtests=true) && ninja

dist: release
	cd build && ninja-dist

vendor/%/README.md:
	@echo "[Fetching submodules]"
	@git submodule update --init
