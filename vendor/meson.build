vendor_cpp_sources = files([
  'edlib/edlib/src/edlib.cpp',

  'rampler/src/sampler.cpp',

  'spoa/src/alignment_engine.cpp',
  'spoa/src/dispatcher.cpp',
  'spoa/src/graph.cpp',
  'spoa/src/simd_alignment_engine_dispatch.cpp',
  'spoa/src/sisd_alignment_engine.cpp',
])

vendor_include_directories = [
                include_directories('edlib/edlib/include'),
                include_directories('rampler/src'),
                include_directories('spoa/include'),
                include_directories('spoa/vendor/cereal/include'),
                include_directories('spoa/vendor/bioparser/include'),
                include_directories('spoa/vendor/bioparser/vendor/biosoup/include'),
                include_directories('thread_pool/include')
                ]

vendor_extra_flags = []

vendor_lib_install = (not meson.is_subproject()) or (get_option('default_library') == 'shared')

vendor_lib = library(
  'vendor',
  vendor_cpp_sources,
  soversion : 0,
  version : meson.project_version(),
  install : vendor_lib_install,
  link_with : [],
  dependencies : [racon_thread_dep, racon_zlib_dep],
  include_directories : vendor_include_directories,
  cpp_args : [vendor_extra_flags, racon_warning_flags, racon_cpp_flags])
